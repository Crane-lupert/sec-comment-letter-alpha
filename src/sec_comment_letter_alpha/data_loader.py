"""SEC UPLOAD/CORRESP loader — daemon-mediated cache reads.

All SEC fetches go through `shared_utils.sec_client`. We never call SEC directly.
Cache miss enqueues a request; the coordination daemon fulfils it and writes
`{cik}_upload-corresp.json` to `<coord_root>/sec-data/edgar-raw/upload-corresp/`.

A cached file shape (from sec-agent-daemon.py):
{
    "cik": "<cik10>",
    "filing_type": "upload-corresp",
    "filings": [
        {"form": "UPLOAD"|"CORRESP", "accession": "...", "date": "YYYY-MM-DD",
         "primary": "...", "text": "<<=500KB>"},
        ...
    ],
    "miss": <bool, only present on miss>
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from shared_utils.sec_client import (
    CacheLookup,
    fetch_from_cache_or_queue,
    iter_cached,
    load_cached,
)


PROJECT_TAG = "X"
FILING_TYPE = "upload-corresp"


EXTRACTABLE_EXTS = {".htm", ".html", ".txt", ".pdf"}
TEXTUAL_EXTS = EXTRACTABLE_EXTS  # backwards-compat alias


@dataclass(frozen=True)
class FilingRecord:
    """One UPLOAD or CORRESP filing extracted from the cache file."""
    cik: str
    form: str
    accession: str
    date: str
    text: str
    primary: str = ""

    @property
    def ext(self) -> str:
        return Path(self.primary).suffix.lower()

    @property
    def is_textual(self) -> bool:
        """True iff `primary` extension yields extractable text (HTML/text/PDF)."""
        return self.ext in EXTRACTABLE_EXTS


def request_filings(cik: str) -> CacheLookup:
    """Cache hit returns path; miss enqueues a fetch request for the daemon."""
    return fetch_from_cache_or_queue(cik, FILING_TYPE, project=PROJECT_TAG)


def request_many(ciks: list[str]) -> dict[str, CacheLookup]:
    return {c: request_filings(c) for c in ciks}


def iter_filings_in_cache(*, textual_only: bool = False) -> Iterator[FilingRecord]:
    """Iterate every cached UPLOAD/CORRESP filing across all CIKs in the cache.

    `textual_only`: if True, skip PDF blobs (the daemon stores raw bytes for these,
    which is mojibake to an LLM). Day 1 dry-run uses textual_only=True.
    """
    for cik_stem, payload in iter_cached(FILING_TYPE):
        cik = payload.get("cik") or cik_stem.split("_")[0]
        for f in payload.get("filings") or []:
            rec = FilingRecord(
                cik=str(cik),
                form=f.get("form", ""),
                accession=f.get("accession", ""),
                date=f.get("date", ""),
                text=f.get("text", ""),
                primary=f.get("primary", ""),
            )
            if textual_only and not rec.is_textual:
                continue
            yield rec


def load_one(path: Path) -> dict:
    return load_cached(path)
