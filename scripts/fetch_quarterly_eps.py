"""Fetch quarterly EPS history for R3K tickers via yfinance.

Output: data/r3k_quarterly_eps.parquet
  Columns: ticker, quarter_end (date), eps_actual

yfinance's `Ticker.income_stmt(freq='quarterly')` exposes 'Basic EPS' or
'Diluted EPS' rows; we use Diluted EPS when available.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R3K = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
OUT = REPO_ROOT / "data" / "r3k_quarterly_eps.parquet"


def _eps_one(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        is_q = tk.quarterly_income_stmt
    except Exception:
        return pd.DataFrame()
    if is_q is None or is_q.empty:
        return pd.DataFrame()
    # Look for diluted EPS row, fallback to basic
    eps_row = None
    for key in ["Diluted EPS", "Basic EPS"]:
        if key in is_q.index:
            eps_row = is_q.loc[key]
            break
    if eps_row is None:
        return pd.DataFrame()
    df = pd.DataFrame({
        "quarter_end": pd.to_datetime(eps_row.index),
        "eps_actual": pd.to_numeric(eps_row.values, errors="coerce"),
    })
    df["ticker"] = ticker
    return df.dropna(subset=["eps_actual"])


def main():
    r3k = pd.read_parquet(R3K)
    tickers = r3k["ticker"].astype(str).unique().tolist()
    print(f"[eps] R3K tickers: {len(tickers)}")
    rows = []
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        rows.append(existing)
        done = set(existing["ticker"].unique())
        print(f"[eps] resuming: {len(done)} already done")
    else:
        done = set()
    todo = [t for t in tickers if t and t not in done]
    print(f"[eps] todo: {len(todo)}")

    started = time.time()
    fails = 0
    for i, t in enumerate(todo):
        try:
            df = _eps_one(t)
            if not df.empty:
                rows.append(df)
        except Exception:
            fails += 1
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            combined.drop_duplicates(subset=["ticker", "quarter_end"], inplace=True)
            combined.to_parquet(OUT, index=False)
            elapsed = time.time() - started
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (len(todo) - (i + 1)) / rate / 60 if rate else float("inf")
            print(f"[eps] {i+1}/{len(todo)}, fails={fails}, rate={rate:.2f}/s, ETA={eta:.1f}min, "
                  f"unique_tickers={combined['ticker'].nunique()}")

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.drop_duplicates(subset=["ticker", "quarter_end"], inplace=True)
    combined.to_parquet(OUT, index=False)
    print(f"[eps] DONE. unique tickers w/ EPS: {combined['ticker'].nunique()}, "
          f"rows: {len(combined)}, date range: {combined['quarter_end'].min()} .. {combined['quarter_end'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
