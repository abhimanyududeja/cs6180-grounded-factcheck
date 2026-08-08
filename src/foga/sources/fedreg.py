"""Federal Register — recent immigration rules and notices.

Official JSON API, no key required:
    GET https://www.federalregister.gov/api/v1/documents.json

Why a fact-checker needs this: 8 CFR and the Policy Manual describe the law as
it currently stands, but they don't tell you what *just changed*. A claim like
"the H-1B registration fee is $10" can be true in a cached corpus and false as
of last month's final rule. We pull the last two years of USCIS and State
Department documents so the system can surface a "recent activity" warning
alongside its verdict.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from urllib.parse import urlencode

from ..config import Config
from ..schema import Document
from .http import PoliteSession

FIELDS = [
    "document_number", "title", "abstract", "publication_date", "type",
    "html_url", "agencies", "effective_on", "action", "citation",
]


def download(cfg: Config, session: PoliteSession, force: bool = False) -> Path:
    api = cfg.get_path("sources.fedreg.api")
    agencies = cfg.get_path("sources.fedreg.agencies", [])
    days = cfg.get_path("sources.fedreg.lookback_days", 730)
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()

    rows: list[dict] = []
    for agency in agencies:
        page = 1
        while page <= 10:  # API caps at 2,000 results; 10 pages x 200 is plenty
            params = [
                ("per_page", "200"),
                ("page", str(page)),
                ("order", "newest"),
                ("conditions[publication_date][gte]", since),
                ("conditions[agencies][]", agency),
            ]
            params += [("fields[]", f) for f in FIELDS]
            url = f"{api}/documents.json?{urlencode(params)}"
            body = session.get(url, suffix=".json", force=force)
            if not body:
                break
            data = json.loads(body)
            batch = data.get("results") or []
            rows.extend(batch)
            if len(batch) < 200:
                break
            page += 1
        print(f"  FedReg: {agency}: {len(rows)} cumulative documents")

    dest = Path(cfg.path("raw")) / "federal_register.json"
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"  FedReg: {len(rows)} documents -> {dest.name}")
    return dest


def parse(cfg: Config, path: Path) -> list[Document]:
    rows = json.loads(path.read_text())
    as_of = dt.date.today().isoformat()

    docs: list[Document] = []
    seen: set[str] = set()
    for r in rows:
        num = r.get("document_number")
        if not num or num in seen:
            continue
        seen.add(num)
        abstract = (r.get("abstract") or "").strip()
        if len(abstract) < 80:
            continue  # metadata-only entries add noise, not evidence
        agencies = ", ".join(a.get("name", "") for a in (r.get("agencies") or []))
        body = (
            f"Document type: {r.get('type', '')}\n"
            f"Action: {r.get('action') or 'n/a'}\n"
            f"Agency: {agencies}\n"
            f"Published: {r.get('publication_date')}\n"
            f"Effective: {r.get('effective_on') or 'not specified'}\n\n"
            f"{abstract}"
        )
        docs.append(
            Document(
                source="fedreg",
                doc_id=f"fedreg:{num}",
                citation=r.get("citation") or f"Fed. Reg. Doc. {num}",
                title=(r.get("title") or "").strip(),
                text=body,
                url=r.get("html_url") or f"https://www.federalregister.gov/d/{num}",
                breadcrumb=["Federal Register", r.get("type", "")],
                as_of=as_of,
                effective_date=r.get("effective_on") or r.get("publication_date"),
            )
        )
    print(f"  FedReg: parsed {len(docs)} documents")
    return docs
