"""One-shot universe bootstrap → data/universe_ciks.parquet.

Builds the Russell-3000-ish CIK list that drives every subsequent enqueue. This
is the ONE place in this repo that does not go through the coordination daemon.

Why the exception:
- The daemon API only fetches FILINGS (per-CIK). It cannot enumerate the universe.
- The portfolio-coordination 'no direct SEC' rule targets the recurring,
  rate-limit-sensitive filing-fetch loop (4 agents × 30K filings).
- This script runs ONCE at setup, single GET on a static metadata endpoint
  (company_tickers.json, ~1.5 MB). It does not race the daemon and does not
  recur in the pipeline.
- Output is a parquet committed to local data/ (gitignored). Pipeline reads only.

Pipeline modules under src/sec_comment_letter_alpha/ stay daemon-only.

Day 2 refinement: replace SEC bootstrap with iShares IWV holdings CSV or
CRSP-Compustat link for the actual Russell 3000 membership; this script's
output is a superset filter, not the final universe.

Run:
    uv run python scripts/bootstrap_universe.py
    uv run python scripts/bootstrap_universe.py --max 1500
"""

from __future__ import annotations

import argparse
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
OUT_PATH = REPO_ROOT / "data" / "universe_ciks.parquet"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _load_user_agent() -> str:
    """Reuse the coord repo's SEC_USER_AGENT so we identify identically to the daemon."""
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(COORD_ROOT / ".env")
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "example.com" in ua:
        raise RuntimeError(
            "SEC_USER_AGENT missing. Set it in either:\n"
            f"  {REPO_ROOT / '.env'}\n"
            f"  {COORD_ROOT / '.env'}\n"
            "Format: 'name email@example.com / purpose'"
        )
    return ua


def fetch_company_tickers(ua: str) -> dict:
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    with httpx.Client(timeout=30) as c:
        r = c.get(SEC_TICKERS_URL, headers=headers)
        r.raise_for_status()
        return r.json()


def to_dataframe(payload: dict, max_n: int | None = None) -> pd.DataFrame:
    rows = []
    for v in payload.values():
        cik = str(v["cik_str"]).zfill(10)
        rows.append({
            "cik": cik,
            "ticker": v.get("ticker", ""),
            "name": v.get("title", ""),
            "source": "sec_company_tickers",
            "bootstrap_ts": datetime.now(timezone.utc).isoformat(),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["cik"]).sort_values("cik").reset_index(drop=True)
    if max_n:
        df = df.head(max_n)
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None, help="cap universe size (sorted by CIK asc)")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args(argv)

    print(f"[bootstrap] fetching {SEC_TICKERS_URL} (one-shot, static metadata)")
    ua = _load_user_agent()
    payload = fetch_company_tickers(ua)
    df = to_dataframe(payload, max_n=args.max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[bootstrap] wrote {len(df)} CIKs → {args.out}")
    print(json.dumps({
        "n_ciks": len(df),
        "first": df.head(3).to_dict(orient="records"),
        "last": df.tail(3).to_dict(orient="records"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
