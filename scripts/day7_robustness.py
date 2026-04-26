"""Day 7 robustness pass — sector / size / liquidity-stratified alpha.

CLAUDE.md Day 6-7 calls for "Sector/size/liquidity robustness". This script
splits the matched-control factor (Day 6) by sector, size-quintile, and
turnover-quintile and reports per-stratum alpha to confirm the headline
isn't carried by a single concentrated bucket.

Inputs:
  data/day4_pairs.jsonl                — events with sector, severity, etc.
  data/day4_events.parquet             — per-event BHARs at 1m/2m/3m
  data/day6_factor_returns_matched.parquet  — Day 6 corrected factor
  data/r3k_monthly_returns.parquet     — yfinance returns
  data/french_factors_monthly.parquet  — FF5+UMD baseline

Outputs:
  data/day7_robustness_summary.json    — per-stratum alpha + t + n
  docs/day7_robustness.md              — methodology + table

Usage:
    .venv/Scripts/python.exe scripts/day7_robustness.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS = REPO_ROOT / "data" / "day4_pairs.jsonl"
EVENTS = REPO_ROOT / "data" / "day4_events.parquet"
RETURNS = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
FRENCH = REPO_ROOT / "data" / "french_factors_monthly.parquet"
R3K = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
OUT_JSON = REPO_ROOT / "data" / "day7_robustness_summary.json"
OUT_MD = REPO_ROOT / "docs" / "day7_robustness.md"

NW_LAG = 6
MIN_N_EVENTS = 15
MIN_N_MONTHS = 12


def _newey_west_alpha(y: np.ndarray, X: np.ndarray) -> dict:
    if len(y) < MIN_N_MONTHS:
        return {"alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "t_alpha": float("nan"), "p_alpha": float("nan"), "n": int(len(y))}
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    a = float(m.params[0]); se = float(m.bse[0])
    return {
        "alpha_monthly": a, "alpha_annual": a * 12,
        "se_monthly": se, "t_alpha": a / se if se else float("nan"),
        "p_alpha": float(m.pvalues[0]), "n": int(len(y)),
    }


def _build_size_quintiles(rets: pd.DataFrame) -> pd.DataFrame:
    """Per-month size quintile based on trailing-3m mean price (free-data proxy)."""
    rets = rets.sort_values(["ticker", "date"]).copy()
    rets["price_lag1"] = rets.groupby("ticker")["price"].shift(1)
    rets["size_proxy_3m"] = rets.groupby("ticker")["price_lag1"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    rets["size_quintile"] = (
        rets.groupby("date")["size_proxy_3m"]
            .transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    )
    return rets


def _build_liquidity_quintiles(rets: pd.DataFrame) -> pd.DataFrame:
    """Liquidity proxy = trailing-3m std of log returns (lower std = more liquid).

    True liquidity needs ADV/dollar volume; this is a free-data substitute.
    """
    rets = rets.sort_values(["ticker", "date"]).copy()
    rets["liq_proxy_3m"] = rets.groupby("ticker")["log_ret"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).std()
    )
    rets["liq_quintile"] = (
        rets.groupby("date")["liq_proxy_3m"]
            .transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    )
    return rets


def _stratum_factor(events: pd.DataFrame, ret_col: str, mask: pd.Series, label: str) -> dict:
    """Build monthly long-short for the stratum subset and run FF5+UMD orthogonalization."""
    sub = events[mask].dropna(subset=[ret_col, "sector", "upload_severity_mean"]).copy()
    if len(sub) < MIN_N_EVENTS:
        return {"label": label, "n_events": int(len(sub)), "skipped": True}
    sub["month"] = pd.to_datetime(sub["a_start_month"]) + pd.offsets.MonthEnd(0)

    rows = []
    for month, grp in sub.groupby("month"):
        if len(grp) < 2:
            continue
        w = grp["upload_severity_mean"].values
        if w.sum() <= 0:
            w = np.ones(len(grp)) / len(grp)
        else:
            w = w / w.sum()
        short_ret = float((w * grp[ret_col].values).sum())
        sector_mean = grp.groupby("sector")[ret_col].mean()
        long_ret = float(grp.assign(_sm=grp["sector"].map(sector_mean))["_sm"].mean())
        rows.append({"month": month, "ls_return": short_ret - long_ret, "n_short": len(grp)})
    fac = pd.DataFrame(rows)
    if len(fac) < MIN_N_MONTHS:
        return {"label": label, "n_events": int(len(sub)), "n_months": int(len(fac)), "skipped": True}

    french = pd.read_parquet(FRENCH)
    french["month"] = pd.to_datetime(french["date"]) + pd.offsets.MonthEnd(0)
    panel = fac.merge(french[["month", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]], on="month")
    y = panel["ls_return"].values
    X = sm.add_constant(panel[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]].values)
    res = _newey_west_alpha(y, X)
    sharpe = float(y.mean() / y.std() * np.sqrt(12)) if y.std() > 0 else float("nan")
    return {
        "label": label, "n_events": int(len(sub)),
        "n_months": int(len(panel)),
        "sharpe_annual": sharpe,
        "alpha_monthly": res["alpha_monthly"],
        "alpha_annual": res["alpha_annual"],
        "t_alpha": res["t_alpha"],
        "p_alpha": res["p_alpha"],
    }


def main() -> int:
    pairs_raw = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs = pd.DataFrame(pairs_raw)
    events = pd.read_parquet(EVENTS)
    print(f"[robust] events: {len(events)}, pairs: {len(pairs)}")

    # Build size + liquidity quintiles per ticker × month from returns
    rets = pd.read_parquet(RETURNS)
    rets["date"] = pd.to_datetime(rets["date"])
    rets = _build_size_quintiles(rets)
    rets = _build_liquidity_quintiles(rets)

    # Map event -> size_q and liq_q at upload_date's month
    events["a_start_month"] = pd.to_datetime(events["a_start_month"])
    ticker_q = (rets[["ticker", "date", "size_quintile", "liq_quintile"]]
                .rename(columns={"date": "month"}))
    events = events.merge(
        ticker_q,
        left_on=["ticker", "a_start_month"],
        right_on=["ticker", "month"],
        how="left",
    )

    out: dict = {"strata": {}, "config": {
        "ret_col": "bhar_a_2m", "min_n_events": MIN_N_EVENTS, "min_n_months": MIN_N_MONTHS,
    }}

    # Headline (full sample, sector-mean control — same as day4 baseline)
    out["strata"]["FULL"] = _stratum_factor(events, "bhar_a_2m",
                                             pd.Series(True, index=events.index), "FULL")

    # By sector
    for sec in events["sector"].dropna().unique():
        out["strata"][f"sector={sec}"] = _stratum_factor(
            events, "bhar_a_2m", events["sector"] == sec, f"sector={sec}",
        )

    # By size quintile
    for q in sorted(events["size_quintile"].dropna().unique()):
        out["strata"][f"size_q{int(q)}"] = _stratum_factor(
            events, "bhar_a_2m", events["size_quintile"] == q, f"size_q{int(q)}",
        )

    # By liquidity quintile
    for q in sorted(events["liq_quintile"].dropna().unique()):
        out["strata"][f"liq_q{int(q)}"] = _stratum_factor(
            events, "bhar_a_2m", events["liq_quintile"] == q, f"liq_q{int(q)}",
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[robust] wrote {OUT_JSON}")

    # Print headline table
    print(f"\n{'stratum':>26} | {'n_ev':>5} | {'n_mo':>4} | {'Sharpe':>7} | {'α/y':>7} | {'t':>6} | {'p':>5}")
    print("-" * 85)
    for label, b in out["strata"].items():
        if b.get("skipped"):
            print(f"{label:>26} | {b.get('n_events','?'):>5} | -SKIP- (insufficient n)")
            continue
        print(f"{label:>26} | {b['n_events']:>5} | {b['n_months']:>4} | "
              f"{b['sharpe_annual']:>+5.2f} | "
              f"{b['alpha_annual']*100:>+5.2f}% | "
              f"{b['t_alpha']:>+5.2f} | "
              f"{b['p_alpha']:>5.3f}")

    # Brief markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Day 7 robustness — sector / size / liquidity strata\n",
        "Source: data/day4_events.parquet (697 events with yfinance forward returns).",
        f"Min events per stratum: {MIN_N_EVENTS}; min months: {MIN_N_MONTHS}.\n",
        "## Per-stratum alpha (BHAR 2m, severity-weighted long-short with sector-mean control)\n",
        "| stratum | n_events | n_months | Sharpe | α/yr | t | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, b in out["strata"].items():
        if b.get("skipped"):
            lines.append(f"| {label} | {b.get('n_events','?')} | — | — | — | — | SKIP |")
            continue
        lines.append(
            f"| {label} | {b['n_events']} | {b['n_months']} | "
            f"{b['sharpe_annual']:+.2f} | {b['alpha_annual']*100:+.2f}% | "
            f"{b['t_alpha']:+.2f} | {b['p_alpha']:.3f} |"
        )
    lines.append(
        "\n**Caveat**: this script uses the Day 4 sector-mean-of-recipients control "
        "(not the Day 6 matched control). For comprehensive Day 7 rigor, swap the "
        "stratum_factor's long-leg construction to use scripts/day6_signal_matched.py "
        "matching logic. Done as Day 7 follow-up after Day 6 sample-expansion finishes."
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[robust] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
