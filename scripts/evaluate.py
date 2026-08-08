#!/usr/bin/env python
"""Step 4: evaluate the system against the gold set.

    python scripts/evaluate.py --retrieval          # free, no API calls
    python scripts/evaluate.py --generation         # full pipeline (costs ~$0.30)
    python scripts/evaluate.py --all
    python scripts/evaluate.py --generation --limit 8      # quick check
    python scripts/evaluate.py --generation --provider ollama   # local ablation

Retrieval evaluation is free and fast, so run it on every config change. The
generation evaluation costs money and takes minutes, so run it when something
structural changed.

Results land in reports/ as JSON, a markdown table for the report, and a chart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.config import load_config  # noqa: E402
from foga.evaluation.metrics import (  # noqa: E402
    format_table,
    score_generation,
    score_retrieval,
)
from foga.factcheck import FactChecker  # noqa: E402
from foga.index import HybridIndex  # noqa: E402
from foga.llm import LLM  # noqa: E402
from foga.retrieve import Retriever  # noqa: E402


def load_goldset(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# Retrieval evaluation + ablation
# ---------------------------------------------------------------------------

def eval_retrieval(index, cfg, gold: list[dict], verbose: bool = True) -> dict:
    """The ablation grid. Each row isolates one design decision so the report
    can say what it actually bought, rather than asserting that hybrid search
    and reranking are good ideas in general."""
    configs = [
        ("dense only",            {"mode": "dense",  "rerank": False, "auth": 0.0}),
        ("BM25 only",             {"mode": "sparse", "rerank": False, "auth": 0.0}),
        ("hybrid (RRF)",          {"mode": "hybrid", "rerank": False, "auth": 0.0}),
        ("hybrid + rerank",       {"mode": "hybrid", "rerank": True,  "auth": 0.0}),
        ("hybrid + rerank + auth", {"mode": "hybrid", "rerank": True,
                                    "auth": cfg.get_path("retrieval.authority_bonus", 0.6)}),
    ]
    rows, detail = [], {}

    for label, opts in configs:
        cfg["retrieval"]["authority_bonus"] = opts["auth"]
        retriever = Retriever(index, cfg, use_reranker=opts["rerank"])
        cases = []
        t0 = time.time()
        for item in gold:
            # Only provision-level labels are scored here. A document-level label
            # ("IRS Publication 501") is satisfied by any chunk of a 200k-character
            # publication, so mixing the two would inflate recall and break
            # comparability with the immigration-only ablation.
            if item.get("citation_granularity", "provision") != "provision":
                continue
            if not item.get("gold_citation"):
                continue
            res = retriever.search(item["claim"], final_k=8, mode=opts["mode"])
            cases.append({
                "id": item["id"],
                "gold_citation": item["gold_citation"],
                "retrieved": [r.chunk.citation for r in res.results],
            })
        m = score_retrieval(cases)
        row = {"config": label, **{k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in m.as_row().items()},
               "sec": round(time.time() - t0, 1)}
        rows.append(row)
        detail[label] = m.per_item
        if verbose:
            print(f"  {label:20s} R@1={m.recall_1:.3f} R@5={m.recall_5:.3f} "
                  f"MRR={m.mrr:.3f}  ({row['sec']}s)")

    return {"table": rows, "detail": detail}


# ---------------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------------

# The three configurations the project proposal calls for: a no-retrieval control,
# a simple-retrieval variant, and the full system. Each is an ablation of the same
# pipeline rather than a separate implementation, so a difference between rows is
# attributable to the setting that changed.
GEN_CONFIGS = {
    "no_retrieval": {
        "retrieval.enabled": False,
        "factcheck.decompose": False,
        "factcheck.verify_citations": False,
    },
    "simple_retrieval": {
        "retrieval.enabled": True,
        "retrieval.mode": "dense",
        "retrieval.rerank": False,
        "retrieval.authority_bonus": 0.0,
        "factcheck.decompose": False,
        # The verifier still RUNS so the baseline reports a real grounding number,
        # it just does not act on it. Measuring only where it is enforced makes an
        # unaudited baseline look perfectly faithful and inverts the comparison.
        "factcheck.verify_citations": False,
    },
    # Isolates decomposition: identical to `full` in every other respect. If this
    # beats `full`, the loss is attributable to splitting the claim, not to hybrid
    # retrieval, reranking or the verifier.
    "full_no_decompose": {
        "retrieval.enabled": True,
        "retrieval.mode": "hybrid",
        "retrieval.rerank": True,
        "factcheck.decompose": False,
        "factcheck.verify_citations": True,
    },
    "full": {
        "retrieval.enabled": True,
        "retrieval.mode": "hybrid",
        "retrieval.rerank": True,
        "factcheck.decompose": True,
        "factcheck.verify_citations": True,
    },
}


def _apply(cfg, overrides: dict) -> None:
    for dotted, value in overrides.items():
        node, _, leaf = dotted.rpartition(".")
        target = cfg
        for part in node.split("."):
            target = target.setdefault(part, {})
        target[leaf] = value


def eval_generation_configs(index, gold, names, provider=None, model=None,
                            verbose=True) -> dict:
    """Run several configurations over the same gold set and compare them."""
    rows, detail = [], {}
    for name in names:
        print(f"\n=== configuration: {name} ===")
        cfg = load_config()
        _apply(cfg, GEN_CONFIGS[name])
        out = eval_generation(index, cfg, gold, provider=provider, model=model,
                              decompose=cfg.get_path("factcheck.decompose", True),
                              verbose=verbose, _cfg_ready=True)
        rows.append({"config": name, **out["metrics"]})
        detail[name] = out
    return {"table": rows, "detail": detail}


def eval_generation(index, cfg, gold: list[dict], provider=None, model=None,
                    decompose=True, verbose=True, _cfg_ready=False) -> dict:
    # When called from eval_generation_configs the cfg already carries the
    # ablation overrides, so reloading it here would silently discard them.
    if not _cfg_ready:
        cfg = load_config()
    cfg["factcheck"]["decompose"] = decompose
    llm = LLM(provider=provider, model=model)
    checker = FactChecker(index, cfg, llm=llm)

    cases, raw = [], []
    for i, item in enumerate(gold, 1):
        t0 = time.time()
        try:
            result = checker.check(item["claim"], verbose=False, currency=False)
        except Exception as exc:
            print(f"  [{i}/{len(gold)}] {item['id']}: ERROR {type(exc).__name__}: {exc}")
            continue
        elapsed = time.time() - t0

        cited = [s["citation"] for sub in result.subclaims for s in sub.sources
                 if s["tag"] in (sub.grounding.get("valid_tags") or [])] or \
                [s["citation"] for sub in result.subclaims for s in sub.sources[:3]]
        fidelities = [sub.grounding.get("quote_fidelity") for sub in result.subclaims
                      if sub.grounding.get("quote_fidelity") is not None]
        grades = [sub.grounding.get("grade") for sub in result.subclaims]

        case = {
            "id": item["id"],
            "gold_verdict": item["gold_verdict"],
            "predicted_verdict": result.verdict,
            "gold_citation": item.get("gold_citation", ""),
            "cited_citations": cited,
            "quote_fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
            "grounding_grade": ("UNGROUNDED" if "UNGROUNDED" in grades
                                else "PARTIAL" if "PARTIAL" in grades else "GROUNDED"),
            "latency_s": elapsed,
            "cost_usd": result.stats.get("est_cost_usd", 0),
        }
        cases.append(case)
        raw.append({**case, "claim": item["claim"], "summary": result.summary,
                    "explanation": result.explanation})

        if verbose:
            ok = "OK  " if result.verdict == item["gold_verdict"] else "MISS"
            print(f"  [{i:2d}/{len(gold)}] {ok} {item['id']:32s} "
                  f"gold={item['gold_verdict']:20s} pred={result.verdict:20s} "
                  f"{case['grounding_grade']}")

    m = score_generation(cases)
    return {"metrics": m.as_row(), "per_item": m.per_item, "raw": raw}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval", action="store_true", help="retrieval metrics + ablation (free)")
    ap.add_argument("--generation", action="store_true", help="full pipeline metrics (costs money)")
    ap.add_argument("--configs", nargs="+", choices=list(GEN_CONFIGS),
                    help="run these configurations and compare them "
                         "(the proposal's no-retrieval and simple-retrieval baselines)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N gold items")
    ap.add_argument("--provider", choices=["openai", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--no-decompose", action="store_true",
                    help="ablate sub-claim decomposition")
    ap.add_argument("--tag", default="", help="label for the output files")
    args = ap.parse_args()

    if args.configs:
        args.generation = True
    if not (args.retrieval or args.generation or args.all):
        args.all = True

    cfg = load_config()
    index = HybridIndex.load(Path(cfg.path("index")))
    gold = load_goldset(Path(cfg.path("eval")) / "goldset.jsonl", args.limit)
    reports = Path(cfg.path("reports"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    tag = f"-{args.tag}" if args.tag else ""

    print(f"index: {len(index):,} chunks ({index.meta.get('embed_model')})")
    print(f"gold set: {len(gold)} claims\n")

    out: dict = {"timestamp": stamp, "index_meta": index.meta, "n_gold": len(gold)}

    if args.retrieval or args.all:
        print("RETRIEVAL ABLATION")
        print("-" * 60)
        out["retrieval"] = eval_retrieval(index, cfg, gold)
        print()
        print(format_table(out["retrieval"]["table"], "Retrieval ablation"))
        print()

    if args.configs:
        print("\nGENERATION EVALUATION - configuration comparison")
        print("-" * 60)
        out["generation_configs"] = eval_generation_configs(
            index, gold, args.configs, provider=args.provider, model=args.model,
        )
        print()
        print(format_table(out["generation_configs"]["table"],
                           "Three-configuration comparison"))
        print()
        print("resid_halluc = ungrounded verdicts still SHOWN as SUPPORTED/CONTRADICTED.")
        print("               The faithfulness number; lower is better.")
        print("detect_halluc = everything the verifier caught, repaired or not. A config")
        print("               with a better verifier scores HIGHER; not a quality ranking.")
    elif args.generation or args.all:
        print("\nGENERATION EVALUATION")
        print("-" * 60)
        out["generation"] = eval_generation(
            index, cfg, gold, provider=args.provider, model=args.model,
            decompose=not args.no_decompose,
        )
        print()
        print(format_table([out["generation"]["metrics"]], "Generation metrics"))
        print()
        misses = [r for r in out.get("generation", {}).get("per_item", []) if not r["correct"]]
        if misses:
            print(f"{len(misses)} misses:")
            for r in misses:
                print(f"  {r['id']:32s} gold={r['gold']:20s} pred={r['pred']}")

    path = reports / f"eval-{stamp}{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwritten: {path}")

    md = reports / f"eval-{stamp}{tag}.md"
    lines = [f"# Evaluation — {stamp}", "",
             f"- index: {len(index):,} chunks, `{index.meta.get('embed_model')}`",
             f"- gold set: {len(gold)} claims", ""]
    if "retrieval" in out:
        lines += ["## Retrieval ablation", "", _md_table(out["retrieval"]["table"]), ""]
    if "generation" in out:
        lines += ["## Generation metrics", "",
                  _md_table([out["generation"]["metrics"]]), ""]
    md.write_text("\n".join(lines))
    print(f"written: {md}")
    return 0


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(none)_"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
