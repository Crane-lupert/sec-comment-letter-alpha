"""Build the actual Russell 3000 universe from iShares IWV holdings.

Replaces the original `bootstrap_universe.py` (which sorted SEC's full ticker
list by CIK ascending, biased to old/established firms — see Day 3 audit).

Pipeline:
  1. Download iShares IWV ETF holdings CSV (public, no cookie wall).
  2. Filter to US-listed equities.
  3. Map ticker -> CIK via SEC's company_tickers.json (one-shot static metadata,
     same exception class as bootstrap_universe.py).
  4. Save to data/universe_ciks_r3k.parquet.

Output schema:
    cik:       str (10-char zero-padded)
    ticker:    str
    name:      str
    sector:    str (GICS sector from iShares)
    source:    "ishares_iwv"
    bootstrap_ts: ISO 8601 UTC

Run:
    .venv/Scripts/python.exe scripts/bootstrap_universe_r3k.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
COORD_ROOT = Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination"))
OUT_PATH = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
RAW_PATH = REPO_ROOT / "data" / "iwv_holdings_raw.csv"

IWV_URL = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf"
    "/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

US_EXCHANGES = {
    "NASDAQ", "New York Stock Exchange Inc.", "NYSE", "NYSE Arca",
    "NYSE MKT LLC", "NYSE American",
    "Cboe BZX formerly known as BATS", "Cboe BZX Exchange Inc.",
}


def _load_user_agent() -> str:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(COORD_ROOT / ".env")
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "example.com" in ua:
        raise RuntimeError("SEC_USER_AGENT missing in .env")
    return ua


def fetch_iwv() -> bytes:
    r = httpx.get(IWV_URL, headers={"User-Agent": "Mozilla/5.0 research"},
                  follow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.content


def parse_iwv(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Ticker,Name,"))
    reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_idx:])))
    return [
        row for row in reader
        if row.get("Ticker") and row.get("Asset Class") == "Equity"
        and row.get("Exchange") in US_EXCHANGES
    ]


def fetch_sec_ticker_map(ua: str) -> dict[str, str]:
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    r = httpx.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    r.raise_for_status()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in r.json().values()
    }


def match_to_cik(holdings: list[dict], ticker_map: dict[str, str]) -> tuple[list[dict], list[str]]:
    matched, unmatched = [], []
    for h in holdings:
        t = h["Ticker"].upper().replace(".", "-").strip()
        cik = (
            ticker_map.get(t)
            or ticker_map.get(t.replace("-", "."))
            or ticker_map.get(t.replace("-", ""))
        )
        if cik:
            matched.append({
                "cik": cik, "ticker": h["Ticker"], "name": h["Name"],
                "sector": h.get("Sector", ""),
            })
        else:
            unmatched.append(h["Ticker"])
    return matched, unmatched


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--raw-cache", type=Path, default=RAW_PATH)
    args = p.parse_args(argv)

    ua = _load_user_agent()
    print(f"[bootstrap-r3k] fetching iShares IWV holdings ({IWV_URL[:80]}...)")
    raw = fetch_iwv()
    args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
    args.raw_cache.write_bytes(raw)
    print(f"[bootstrap-r3k] saved raw csv: {args.raw_cache} ({len(raw)} bytes)")

    holdings = parse_iwv(raw)
    print(f"[bootstrap-r3k] parsed {len(holdings)} US-listed equity holdings")

    ticker_map = fetch_sec_ticker_map(ua)
    print(f"[bootstrap-r3k] SEC ticker->CIK map: {len(ticker_map)} entries")

    matched, unmatched = match_to_cik(holdings, ticker_map)
    print(f"[bootstrap-r3k] matched={len(matched)}  unmatched={len(unmatched)}")
    if unmatched:
        print(f"[bootstrap-r3k] unmatched sample: {unmatched[:8]}")

    df = pd.DataFrame(matched)
    df["source"] = "ishares_iwv"
    df["bootstrap_ts"] = datetime.now(timezone.utc).isoformat()
    df = df.drop_duplicates(subset=["cik"]).sort_values("cik").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[bootstrap-r3k] wrote {args.out} -- rows={len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
