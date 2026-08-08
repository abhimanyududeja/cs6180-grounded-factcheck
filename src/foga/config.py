"""Config loading. Every entry point calls `load_config()` so there is exactly
one place that knows what the system is set to do."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict with dotted access: cfg.get_path("retrieval.final_k")."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Resolve a `paths.*` entry to an absolute Path, creating it."""
        p = ROOT / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


_cached: Config | None = None


def load_config(path: str | Path | None = None) -> Config:
    global _cached
    if _cached is not None and path is None:
        return _cached
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as fh:
        cfg = Config(yaml.safe_load(fh))

    # The User-Agent carries a contact address, which must belong to whoever is
    # actually running the downloads — not to whoever wrote the config.
    ua = cfg.get_path("sources.uscis_pm.user_agent", "")
    if "{contact_email}" in ua:
        email = os.environ.get("FOGA_CONTACT_EMAIL", "").strip()
        if not email:
            raise RuntimeError(
                "Set FOGA_CONTACT_EMAIL in .env to your own email address.\n"
                "Federal sites ask automated clients to identify themselves, and "
                "the address must be yours."
            )
        cfg["sources"]["uscis_pm"]["user_agent"] = ua.format(contact_email=email)

    if path is None:
        _cached = cfg
    return cfg


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing {name}. Copy .env.example to .env and fill it in, "
            f"or run: export {name}=..."
        )
    return val
