"""Day 5 PEAD baseline factor — Foster-Olsen-Shevlin (1984) naive surprise.

Surprise(i, q) = (EPS_q - EPS_q-4) / std_dev(EPS_q-1 .. EPS_q-8 minus their lag-4)

Each calendar month, sort firms by latest reported surprise into quintiles;
long top minus short bottom, equal-weight, 1-month hold.

Output: data/day5_pead_factor.parquet (monthly long-short return series)

Note: this is the _naive_ PEAD; the canonical IBES-based version requires
analyst forecasts (paid data). We document this in docs/limitations.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = REPO_ROOT / "data" / "r3k_quarterly_eps.parquet"
RETS = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
OUT = REPO_ROOT / "data" / "day5_pead_factor.parquet"


def _compute_surprises(eps: pd.DataFrame) -> pd.DataFrame:
    """For each ticker, compute SUE = (Q_eps - Q_lag4_eps) / std_dev(8 prior changes)."""
    eps = eps.sort_values(["ticker", "quarter_end"]).copy()
    eps["lag4_eps"] = eps.groupby("ticker")["eps_actual"].shift(4)
    eps["change"] = eps["eps_actual"] - eps["lag4_eps"]
    # Rolling std of change over prior 8 quarters (excluding current)
    eps["rolling_std"] = (
        eps.groupby("ticker")["change"]
           .shift(1)
           .rolling(8, min_periods=4)
           .std()
           .reset_index(0, drop=True)
    )
    eps["sue"] = eps["change"] / eps["rolling_std"]
    return eps[["ticker", "quarter_end", "eps_actual", "change", "sue"]].dropna(subset=["sue"])


def _build_factor(surprises: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Build monthly long-short factor return series.

    For each calendar month M:
      - Restrict to surprises whose quarter_end is in (M-90 days, M] (the
        latest publicly known surprise per ticker).
      - Sort firms into quintiles by SUE.
      - Long top quintile, short bottom quintile, equal-weight.
      - Return = next month's log return of long minus short.
    """
    surprises = surprises.sort_values(["ticker", "quarter_end"]).copy()
    rets = rets.sort_values(["ticker", "date"]).copy()

    months = pd.date_range(rets["date"].min(), rets["date"].max(), freq="ME")
    rows = []
    for m in months:
        cutoff = m
        latest = (surprises[surprises["quarter_end"] <= cutoff]
                  .sort_values("quarter_end")
                  .groupby("ticker", as_index=False).tail(1))
        # Window: surprises within last 90 days (announcement is usually ~30-60d post)
        window_lo = cutoff - pd.Timedelta(days=120)
        latest = latest[latest["quarter_end"] >= window_lo]
        if len(latest) < 30:
            continue
        latest["quintile"] = pd.qcut(latest["sue"], 5, labels=False, duplicates="drop")
        long_t = latest.loc[latest["quintile"] == 4, "ticker"].tolist()
        short_t = latest.loc[latest["quintile"] == 0, "ticker"].tolist()
        # Forward 1m return: month m+1 (i.e. the month following cutoff)
        next_month = (cutoff + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)
        next_rets = rets[rets["date"] == next_month]
        if next_rets.empty:
            continue
        long_ret = next_rets[next_rets["ticker"].isin(long_t)]["log_ret"].mean()
        short_ret = next_rets[next_rets["ticker"].isin(short_t)]["log_ret"].mean()
        if pd.isna(long_ret) or pd.isna(short_ret):
            continue
        rows.append({
            "month": next_month, "ls_return": float(long_ret - short_ret),
            "long_return": float(long_ret), "short_return": float(short_ret),
            "n_long": len(long_t), "n_short": len(short_t),
        })
    return pd.DataFrame(rows)


def main() -> int:
    if not EPS.exists():
        print(f"[pead] {EPS} missing; run scripts/fetch_quarterly_eps.py first")
        return 1
    eps = pd.read_parquet(EPS)
    rets = pd.read_parquet(RETS)
    rets["date"] = pd.to_datetime(rets["date"])
    print(f"[pead] eps rows: {len(eps)}, tickers: {eps['ticker'].nunique()}")
    surprises = _compute_surprises(eps)
    print(f"[pead] surprises: {len(surprises)}, tickers: {surprises['ticker'].nunique()}")
    factor = _build_factor(surprises, rets)
    print(f"[pead] factor months: {len(factor)}")
    if not factor.empty:
        factor.to_parquet(OUT, index=False)
        print(f"[pead] wrote {OUT}")
        sharpe = factor['ls_return'].mean() / factor['ls_return'].std() * np.sqrt(12)
        print(f"  raw sharpe: {sharpe:.2f}, mean monthly: {factor['ls_return'].mean()*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
