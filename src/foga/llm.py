"""LLM access layer.

Two drivers behind one interface:
  * `openai`  — the default. Current model family is GPT-5.6 (sol/terra/luna).
  * `ollama`  — a fully local fallback so the demo still runs with no internet
                and no API key. On an 8 GB machine keep to a ~4B model.

The driver is chosen in config.yaml (`llm.provider`) and can be overridden
per-call, which is what the evaluation ablation uses to compare a small local
model against a frontier model on the exact same retrieval context.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import load_config, require_env

# Params that newer reasoning-tuned models may reject. If the API complains
# about one, we drop it and retry rather than hard-failing the pipeline.
_OPTIONAL_PARAMS = ("temperature", "top_p", "presence_penalty", "frequency_penalty")


@dataclass
class Usage:
    """Token accounting so the report can state exactly what the project cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, i: int, o: int) -> None:
        self.input_tokens += i
        self.output_tokens += o
        self.calls += 1


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)

    def json(self) -> Any:
        """Parse the response as JSON, tolerating markdown fences and preamble."""
        return parse_json_loose(self.text)


def parse_json_loose(text: str) -> Any:
    """Models occasionally wrap JSON in ```json fences or add a sentence before
    it. Recover the payload rather than crashing the pipeline."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace/bracket pair.
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove reasoning blocks a hybrid model may emit before its answer."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    # An unterminated block means the model was cut off mid-reasoning; keep what
    # follows the opening tag rather than returning the reasoning as the answer.
    if "<think>" in cleaned.lower():
        cleaned = re.split(r"</?think>", cleaned, flags=re.IGNORECASE)[-1].strip()
    return cleaned


def _ollama_messages(msgs: list[dict]) -> list[dict]:
    """Convert assistant tool-call arguments back to objects for Ollama.

    Tool-call arguments are a JSON *string* in OpenAI's schema and an *object* in
    Ollama's. The agent loop is written against OpenAI's shape, so a call normalized
    on the way in has to be converted back on the way out. Sent as a string, Ollama
    fails to parse its own chat template and rejects the request with a 400 whose
    message ("Value looks like object, but can't find closing '}' symbol") points at
    JSON rather than at the round-trip that caused it.
    """
    out = []
    for m in msgs:
        calls = m.get("tool_calls")
        if not calls:
            out.append(m)
            continue
        fixed = []
        for c in calls:
            fn = dict(c.get("function", {}))
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args or "{}")
                except json.JSONDecodeError:
                    fn["arguments"] = {}
            fixed.append({**c, "function": fn})
        out.append({**m, "tool_calls": fixed})
    return out


class LLM:
    """Thin, provider-agnostic chat wrapper."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.provider = provider or cfg.get_path("llm.provider", "openai")
        self.temperature = (
            temperature if temperature is not None
            else cfg.get_path("llm.temperature", 0.0)
        )
        self.max_tokens = cfg.get_path("llm.max_output_tokens", 2000)
        self.usage = Usage()
        self._unsupported: set[str] = set()

        if self.provider == "openai":
            from openai import OpenAI

            self.model = model or cfg.get_path("llm.model", "gpt-5.6-luna")
            self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        elif self.provider == "ollama":
            self.model = model or cfg.get_path("llm.ollama_model", "qwen3:4b")
            self.host = cfg.get_path("llm.ollama_host", "http://localhost:11434")
        else:
            raise ValueError(f"Unknown llm provider: {self.provider}")

    # -- public -----------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        schema: dict | None = None,
    ) -> LLMResponse | dict:
        """Single completion. If `tools` is passed, returns the raw OpenAI
        message dict so the caller can inspect tool_calls."""
        msgs = messages or [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.provider == "openai":
            return self._openai(msgs, json_mode=json_mode, tools=tools)
        return self._ollama(msgs, json_mode=json_mode, schema=schema, tools=tools)

    # -- drivers ----------------------------------------------------------

    def _openai(
        self, msgs: list[dict], json_mode: bool, tools: list[dict] | None
    ) -> LLMResponse | dict:
        from openai import BadRequestError

        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs}
        if "temperature" not in self._unsupported:
            kwargs["temperature"] = self.temperature
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            # Reasoning models default to a non-zero reasoning_effort, and
            # /v1/chat/completions rejects that combined with function tools. The
            # API's own advice is to set it to "none". It is not otherwise in
            # kwargs, so the retry loop below cannot recover by dropping it: the
            # parameter has to be added, not removed.
            if "reasoning_effort" not in self._unsupported:
                kwargs["reasoning_effort"] = "none"

        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except BadRequestError as exc:
                # Drop whichever optional param the model rejected, then retry.
                dropped = next(
                    (p for p in _OPTIONAL_PARAMS if p in kwargs and p in str(exc)), None
                )
                if dropped:
                    self._unsupported.add(dropped)
                    kwargs.pop(dropped)
                    continue
                if "max_tokens" in str(exc) and "max_completion_tokens" not in kwargs:
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = self.max_tokens
                    continue
                raise
            except Exception as exc:  # rate limit / transient network
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        else:  # pragma: no cover
            raise RuntimeError("OpenAI call failed after retries")

        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        if tools:
            return msg.model_dump()
        return LLMResponse(text=msg.content or "", model=self.model, usage=self.usage)

    def _ollama(self, msgs: list[dict], json_mode: bool,
                schema: dict | None = None,
                tools: list[dict] | None = None) -> LLMResponse | dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": self.temperature},
            # qwen3 and other hybrid reasoning models emit a long <think> block
            # before the answer unless told not to. Left on it costs minutes per
            # call and, worse, the reasoning text bleeds into quoted spans, so the
            # mechanical quote check fails on answers that were actually correct.
            "think": False,
        }
        if tools:
            payload["tools"] = tools
            payload["messages"] = _ollama_messages(msgs)
            # Tool calling and constrained decoding are mutually exclusive: a schema
            # would force the model to emit an answer object when it should be
            # emitting a tool call.
            payload.pop("format", None)
        if schema is not None and not tools:
            # Structured outputs: constrain decoding to the schema, not merely to
            # valid JSON. This is what stops a small model returning well-formed
            # output with entirely different keys.
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}"
            )
        if not r.ok:
            # Ollama puts the reason in the body; raise_for_status alone reports only
            # the status code, which makes a malformed message list impossible to debug.
            raise RuntimeError(
                f"Ollama {r.status_code} for model {self.model}: {r.text[:400]}"
            )
        data = r.json()
        self.usage.add(
            data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        )
        message = data["message"]

        if tools:
            # The agent loop is written against OpenAI's message shape. Ollama
            # returns arguments as a dict where OpenAI returns a JSON string, and
            # omits the call id entirely, so both are normalized here rather than
            # branching on provider inside the agent.
            calls = []
            for i, c in enumerate(message.get("tool_calls") or []):
                fn = c.get("function", {})
                args = fn.get("arguments")
                calls.append({
                    "id": c.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                    },
                })
            return {
                "role": "assistant",
                "content": _strip_think(message.get("content") or ""),
                "tool_calls": calls or None,
            }

        # Defensive: `think: False` is ignored by models that do not support it,
        # and a stray reasoning block would otherwise be parsed as the answer.
        return LLMResponse(
            text=_strip_think(message["content"]), model=self.model,
            usage=self.usage,
        )


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

class Embedder:
    """OpenAI or local sentence-transformers embeddings, same interface."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        cfg = load_config()
        self.provider = provider or cfg.get_path("retrieval.embed_provider", "openai")
        self.batch_size = cfg.get_path("retrieval.embed_batch_size", 128)

        if self.provider == "openai":
            from openai import OpenAI

            self.model = model or cfg.get_path(
                "retrieval.embed_model", "text-embedding-3-small"
            )
            self.dims = cfg.get_path("retrieval.embed_dims", 1536)
            self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        else:
            from sentence_transformers import SentenceTransformer

            self.model = model or cfg.get_path(
                "retrieval.local_embed_model", "BAAI/bge-small-en-v1.5"
            )
            self.st = SentenceTransformer(self.model)
            self.dims = self.st.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = False):
        import numpy as np

        if self.provider != "openai":
            vecs = self.st.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return np.asarray(vecs, dtype="float32")

        out: list[list[float]] = []
        rng = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm

            rng = tqdm(list(rng), desc="embedding", unit="batch")
        for i in rng:
            batch = [t.replace("\n", " ")[:8000] for t in texts[i : i + self.batch_size]]
            for attempt in range(5):
                try:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
            out.extend([d.embedding for d in resp.data])
        arr = np.asarray(out, dtype="float32")
        # Cosine similarity via inner product requires unit vectors.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


def estimate_cost(usage: Usage, model: str) -> float:
    """Rough USD estimate for the report. Prices per 1M tokens, Aug 2026."""
    table = {
        "gpt-5.6-sol": (5.00, 30.00),
        "gpt-5.6-terra": (2.00, 12.00),
        "gpt-5.6-luna": (0.20, 1.20),
        "gpt-5-mini": (0.25, 2.00),
        "gpt-5-nano": (0.05, 0.40),
        "text-embedding-3-small": (0.02, 0.0),
        "text-embedding-3-large": (0.13, 0.0),
    }
    pin, pout = table.get(model, (0.0, 0.0))
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000
