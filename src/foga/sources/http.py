"""Polite HTTP with an on-disk cache.

Three things matter here:

1. **We identify ourselves.** Every request carries a descriptive User-Agent
   with a contact address, which is what federal sites ask automated clients
   to do.
2. **We fetch each URL exactly once, ever.** Responses are cached to
   `data/raw/_cache/`, so re-running the pipeline, restarting after a crash,
   or rebuilding the index costs zero additional requests to the government.
3. **TLS is always verified.** See `_enable_system_trust` below — several
   federal sites need help here, and the fix is to supply the missing
   certificate, never to turn verification off.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from ..config import ROOT

CACHE = ROOT / "data" / "raw" / "_cache"

_trust_ready = False


def _enable_system_trust() -> None:
    """fam.state.gov (and some other .gov hosts) serve only their leaf
    certificate and omit the Sectigo intermediate. Browsers and curl recover by
    following the certificate's Authority Information Access URI to fetch the
    missing intermediate; Python's bundled certifi store does not do this, so
    the handshake fails with 'unable to get local issuer certificate'.

    `truststore` routes verification through the operating system's own trust
    engine (macOS Security.framework / Windows CryptoAPI), which does perform
    that AIA fetch. Verification stays fully on — we are fixing a chain-building
    gap, not disabling a security check. Never replace this with verify=False.
    """
    global _trust_ready
    if _trust_ready:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - non-macOS/Windows or missing package
        pass
    _trust_ready = True


class PoliteSession:
    """A requests.Session that respects a fixed inter-request delay and caches
    every response body on disk."""

    def __init__(self, user_agent: str, delay: float = 1.0, cache_dir: Path | None = None):
        _enable_system_trust()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})
        self.delay = delay
        self.cache = cache_dir or CACHE
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self.stats = {"hits": 0, "fetched": 0, "errors": 0}

    def _cache_path(self, url: str, suffix: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:20]
        return self.cache / f"{h}{suffix}"

    def get(self, url: str, suffix: str = ".html", force: bool = False) -> str | None:
        """Return response text, from cache when available. None on failure."""
        cp = self._cache_path(url, suffix)
        if cp.exists() and not force:
            self.stats["hits"] += 1
            return cp.read_text(encoding="utf-8", errors="ignore")

        wait = self.delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)

        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=60)
                self._last = time.time()
                if r.status_code == 404:
                    self.stats["errors"] += 1
                    return None
                r.raise_for_status()
                cp.write_text(r.text, encoding="utf-8")
                self.stats["fetched"] += 1
                return r.text
            except requests.RequestException:
                if attempt == 2:
                    self.stats["errors"] += 1
                    return None
                time.sleep(5 * (attempt + 1))
        return None

    def get_bytes(self, url: str, dest: Path, force: bool = False) -> Path | None:
        """Stream a binary file (a zip) to `dest`, skipping if already there."""
        if dest.exists() and dest.stat().st_size > 0 and not force:
            self.stats["hits"] += 1
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.session.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for block in r.iter_content(1 << 16):
                        fh.write(block)
            self.stats["fetched"] += 1
            return dest
        except requests.RequestException:
            self.stats["errors"] += 1
            return None
