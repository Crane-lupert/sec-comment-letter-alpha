"""Download monthly adjusted-close returns for R3K tickers via yfinance.

Notes & caveats:
  - yfinance pulls from Yahoo Finance; survivorship bias is built in (delisted
    tickers may return empty data). For HF-grade analysis we'd use CRSP via
    WRDS; this is a v1 free-tier substitute.
  - Some tickers fail (renames, delisting, ETF reclassification). Failures
    are logged but do not abort the run.
  - Output: long-format parquet, one row per (ticker, month-end).

Resumable: skips tickers already in the output parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R3K = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
OUT = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
LOG = REPO_ROOT / "data" / "fetch_yfinance_returns.log"

START = "2014-12-01"  # need t-1 for first 2015 month return
END = "2025-12-31"


def _fetch_one(ticker: str, start: str, end: str):
    import yfinance as yf
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, interval="1mo", auto_adjust=True)
    if hist.empty:
        return pd.DataFrame()
    df = hist[["Close"]].rename(columns={"Close": "price"}).reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None) + pd.offsets.MonthEnd(0)
    df = df[["date", "price"]]
    df["ticker"] = ticker
    df["log_ret"] = (df["price"] / df["price"].shift(1)).apply(lambda x: pd.NA if pd.isna(x) or x <= 0 else float(__import__("math").log(x)))
    return df.dropna(subset=["log_ret"]).reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--limit", type=int, default=None, help="limit ticker count for testing")
    args = p.parse_args(argv)

    r3k = pd.read_parquet(R3K)
    tickers = r3k["ticker"].astype(str).tolist()
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"[yfin] R3K tickers to fetch: {len(tickers)}")

    # Resume support
    done_tickers: set[str] = set()
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        done_tickers = set(existing["ticker"].astype(str).unique())
        print(f"[yfin] resuming: {len(done_tickers)} tickers already done")

    todo = [t for t in tickers if t and t not in done_tickers]
    print(f"[yfin] to-do: {len(todo)}")

    rows: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    if OUT.exists():
        rows.append(pd.read_parquet(OUT))

    started = time.time()
    log_lines = []
    for i, tk in enumerate(todo):
        try:
            df = _fetch_one(tk, args.start, args.end)
            if df.empty:
                failures.append((tk, "empty_history"))
                log_lines.append(f"EMPTY {tk}")
            else:
                rows.append(df)
        except Exception as e:
            failures.append((tk, f"{type(e).__name__}: {e}"))
            log_lines.append(f"FAIL  {tk}: {type(e).__name__}: {e}")
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            # checkpoint write
            combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            combined.drop_duplicates(subset=["ticker", "date"], inplace=True)
            combined.to_parquet(OUT, index=False)
            elapsed = time.time() - started
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (len(todo) - (i + 1)) / rate / 60 if rate else float("inf")
            print(f"[yfin] {i+1}/{len(todo)}, fails={len(failures)}, rate={rate:.2f}/s, ETA={eta:.1f}min")
            LOG.write_text("\n".join(log_lines), encoding="utf-8")

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.drop_duplicates(subset=["ticker", "date"], inplace=True)
    combined.to_parquet(OUT, index=False)
    LOG.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[yfin] DONE.")
    print(f"  total rows: {len(combined)}")
    print(f"  unique tickers w/ data: {combined['ticker'].nunique()}")
    print(f"  failures: {len(failures)}")
    print(f"  date range: {combined['date'].min()} .. {combined['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
