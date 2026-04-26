"""Build the firm-month panel + per-event forward returns.

Inputs:
  - data/r3k_monthly_returns.parquet   (yfinance R3K returns)
  - data/french_factors_monthly.parquet (FF5 + UMD + RF)
  - data/day4_pairs.jsonl              (UPLOAD-CORRESP pairs with features)
  - data/universe_ciks_r3k.parquet     (CIK -> ticker, sector)

Outputs:
  - data/day4_panel.parquet            (firm-month panel)
  - data/day4_events.parquet           (per-event row: pair_id + signal-A and
                                        signal-B forward returns at multiple
                                        horizons, computed without leakage)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RETURNS = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
FRENCH = REPO_ROOT / "data" / "french_factors_monthly.parquet"
PAIRS = REPO_ROOT / "data" / "day4_pairs.jsonl"
R3K = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"

OUT_PANEL = REPO_ROOT / "data" / "day4_panel.parquet"
OUT_EVENTS = REPO_ROOT / "data" / "day4_events.parquet"

# Forward-return horizons in months (calendar months -- yfinance is monthly)
HORIZONS = [1, 2, 3]  # 1m, 2m, 3m forward (rough proxies for 30/60/90 day BHAR)


def _bhar(ret_panel: pd.DataFrame, mkt: pd.DataFrame,
          start_month: pd.Timestamp, n_months: int, ticker: str) -> float | None:
    """Buy-and-hold abnormal return: cum log ret of firm minus cum log ret of market
    over the n-month window starting `start_month` (inclusive)."""
    end_month = start_month + pd.DateOffset(months=n_months - 1)
    end_month = end_month + pd.offsets.MonthEnd(0)
    firm = ret_panel[(ret_panel["ticker"] == ticker)
                     & (ret_panel["date"] >= start_month)
                     & (ret_panel["date"] <= end_month)]["log_ret"]
    if len(firm) < n_months:
        return None
    mkt_w = mkt[(mkt["date"] >= start_month) & (mkt["date"] <= end_month)]["mkt_log_ret"]
    if len(mkt_w) < n_months:
        return None
    return float(firm.sum() - mkt_w.sum())


def _car(ret_panel: pd.DataFrame, mkt: pd.DataFrame,
         start_month: pd.Timestamp, n_months: int, ticker: str) -> float | None:
    """Cumulative abnormal return: sum of (firm - market) per month."""
    end_month = start_month + pd.DateOffset(months=n_months - 1)
    end_month = end_month + pd.offsets.MonthEnd(0)
    firm = ret_panel[(ret_panel["ticker"] == ticker)
                     & (ret_panel["date"] >= start_month)
                     & (ret_panel["date"] <= end_month)][["date", "log_ret"]]
    mkt_w = mkt[(mkt["date"] >= start_month) & (mkt["date"] <= end_month)][["date", "mkt_log_ret"]]
    merged = firm.merge(mkt_w, on="date", how="inner")
    if len(merged) < n_months:
        return None
    return float((merged["log_ret"] - merged["mkt_log_ret"]).sum())


def main() -> int:
    rets = pd.read_parquet(RETURNS)
    rets["date"] = pd.to_datetime(rets["date"])
    print(f"[panel] returns: {len(rets)} rows, {rets['ticker'].nunique()} tickers, "
          f"{rets['date'].min()} .. {rets['date'].max()}")

    french = pd.read_parquet(FRENCH)
    french["date"] = pd.to_datetime(french["date"])
    # Approximate market log return = Mkt-RF + RF (i.e. raw market)
    french["mkt_log_ret"] = french["Mkt-RF"] + french["RF"]
    mkt = french[["date", "mkt_log_ret"]].copy()

    pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[panel] pairs: {len(pairs)}")

    r3k = pd.read_parquet(R3K)
    cik_meta = {row["cik"]: row for _, row in r3k.iterrows()}

    # Compute forward returns per pair, two signals
    rows = []
    drops_no_returns = 0
    for p in pairs:
        cik = p["cik"]
        ticker = p["ticker"]
        upload_date = pd.to_datetime(p["upload_date"])
        corresp_date = pd.to_datetime(p["corresp_date"])
        # Signal A start: month following upload_date
        a_start = (upload_date + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)
        # Signal B start: month following corresp_date
        b_start = (corresp_date + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)

        row = dict(p)  # copy all features
        row["pair_id"] = p["pair_id"]
        row["a_start_month"] = a_start
        row["b_start_month"] = b_start

        any_data = False
        for h in HORIZONS:
            ba = _bhar(rets, mkt, a_start, h, ticker)
            bb = _bhar(rets, mkt, b_start, h, ticker)
            ca = _car(rets, mkt, a_start, h, ticker)
            cb = _car(rets, mkt, b_start, h, ticker)
            row[f"bhar_a_{h}m"] = ba
            row[f"bhar_b_{h}m"] = bb
            row[f"car_a_{h}m"] = ca
            row[f"car_b_{h}m"] = cb
            if ba is not None or bb is not None:
                any_data = True
        if not any_data:
            drops_no_returns += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[panel] events with at least 1 horizon return: {len(df)} (dropped {drops_no_returns} with no yfinance match)")
    OUT_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_EVENTS, index=False)
    print(f"[panel] wrote {OUT_EVENTS}")

    # Coverage report
    for h in HORIZONS:
        for tag in ("bhar_a", "bhar_b"):
            col = f"{tag}_{h}m"
            n = df[col].notna().sum()
            print(f"  {col}: n={n} mean={df[col].mean():.4f} std={df[col].std():.4f}")

    # Build firm-month panel (returns + french factors merged)
    panel = rets.merge(french, on="date", how="left")
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"[panel] wrote {OUT_PANEL} -- rows={len(panel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
