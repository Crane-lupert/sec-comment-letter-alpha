"""Day 7 robustness — sector / size / liquidity-stratified alpha on the
matched-control (Day 6) AND risk-managed (Day 7 N=4) baselines.

The pre-registered `scripts/day7_robustness.py` actually uses the Day 4
sector-mean-of-recipients control (despite its docstring referencing the
Day 6 matched parquet). It is preserved unchanged. This script ADDS two
new robustness summaries on the headline-consistent baselines:

  --input matched   : data/day7_robustness_matched_summary.json
  --input rm        : data/day7_robustness_rm_summary.json

For each stratum we rebuild a monthly long-short factor:

  matched : per-month sev-weighted short(recipient_ret) - eq-weight
            long(matched-control mean BHAR) on the events in that
            stratum. Same matching as scripts/day6_signal_matched.py.

  rm      : same per-event panel as matched, but per-stratum we apply
            the same overlays as scripts/day7_risk_managed_overlay.py:
              A. breadth filter (n_kept >= BREADTH_MIN)
              B. per-name cap = max(0.20, 1.5/N)
              C. 10% annual vol target via 6m lag-1 rolling std,
                 leverage cap = 2.0.

Both then run FF5+UMD orthogonalisation with Newey-West HAC SE (lag=6).

Stratification reuses scripts/day7_robustness.py logic (size & liquidity
quintiles from r3k monthly returns, sector from events). MIN_N_EVENTS=15,
MIN_N_MONTHS=12.

Usage:
    .venv/Scripts/python.exe scripts/day7_robustness_extended.py --input matched
    .venv/Scripts/python.exe scripts/day7_robustness_extended.py --input rm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

REPO_ROOT_LOCAL = _HERE.parents[1]
REPO_ROOT_MAIN = Path("D:/vscode/sec-comment-letter-alpha")


def _pick_data_dir() -> Path:
    for c in [REPO_ROOT_LOCAL / "data", REPO_ROOT_MAIN / "data"]:
        if (c / "day6_factor_returns_matched.parquet").exists():
            return c
    raise SystemExit("day6_factor_returns_matched.parquet not found")


DATA = _pick_data_dir()
REPO_ROOT_USED = DATA.parent

# Reuse Day 6 match logic (point its module-level paths at the resolved DATA).
import day6_signal_matched as _d6  # noqa: E402

_d6.REPO_ROOT = REPO_ROOT_USED
_d6.EVENTS = DATA / "day4_events.parquet"
_d6.PAIRS = DATA / "day4_pairs.jsonl"
_d6.RETURNS = DATA / "r3k_monthly_returns.parquet"
_d6.UNIVERSE = DATA / "universe_ciks_r3k.parquet"

from day6_signal_matched import (  # noqa: E402
    _match_event,
    _bhar_window,
    _car_window,
    _load_french_market,
    _load_letter_dates,
    _build_size_proxy,
    _build_log_ret_pivot,
    _build_letter_recipients_per_month,
)

# Reuse RM overlay knobs and primitives.
from day7_risk_managed_overlay import (  # noqa: E402
    build_per_event_rows,
    cap_weights,
    BREADTH_MIN,
    NAME_CAP_BASE,
    NAME_CAP_INV_N,
    VOL_TARGET_MONTHLY,
    VOL_LOOKBACK,
    LEV_CAP,
)

# Reuse the same SIGNAL_SPECS as day6_signal_matched (one entry per signal id).
from day6_signal_matched import SIGNAL_SPECS  # noqa: E402

# Robustness constants (same spirit as scripts/day7_robustness.py).
NW_LAG = 6
MIN_N_EVENTS = 15
MIN_N_MONTHS = 12

# Headline signal: A_bhar_2m (matched) -> A_bhar_2m_matched / A_bhar_2m_matched_rm
HEADLINE_SPEC = next(s for s in SIGNAL_SPECS if s["signal_id"] == "A_bhar_2m_matched")
HEADLINE_RET_LABEL = "bhar_a_2m"


# --- size / liquidity quintiles (verbatim from day7_robustness.py) ---------

def _build_size_quintiles(rets: pd.DataFrame) -> pd.DataFrame:
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
    rets = rets.sort_values(["ticker", "date"]).copy()
    rets["liq_proxy_3m"] = rets.groupby("ticker")["log_ret"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).std()
    )
    rets["liq_quintile"] = (
        rets.groupby("date")["liq_proxy_3m"]
            .transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    )
    return rets


def _newey_west_alpha(y: np.ndarray, X: np.ndarray) -> dict:
    if len(y) < MIN_N_MONTHS:
        return {"alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "t_alpha": float("nan"), "p_alpha": float("nan"),
                "se_monthly": float("nan"), "n": int(len(y))}
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    a = float(m.params[0]); se = float(m.bse[0])
    return {
        "alpha_monthly": a, "alpha_annual": a * 12,
        "se_monthly": se, "t_alpha": a / se if se else float("nan"),
        "p_alpha": float(m.pvalues[0]), "n": int(len(y)),
    }


def _aggregate_matched_monthly(per_event: pd.DataFrame) -> pd.DataFrame:
    """Severity-weighted short minus eq-weight long, per month."""
    rows = []
    for month, grp in per_event.groupby("month"):
        if len(grp) < 2:
            continue
        sev = grp["severity"].to_numpy().astype(float)
        if sev.sum() <= 0:
            w = np.ones(len(sev)) / len(sev)
        else:
            w = sev / sev.sum()
        short_ret = float((w * grp["recipient_ret"].to_numpy()).sum())
        long_ret = float(grp["control_ret"].mean())
        rows.append({
            "month": pd.Timestamp(month),
            "ls_return": short_ret - long_ret,
            "n_events": int(len(grp)),
        })
    return pd.DataFrame(rows)


def _aggregate_rm_monthly(per_event: pd.DataFrame) -> pd.DataFrame:
    """Apply RM overlays A+B per month, then C across the time series."""
    pre_rows = []
    for month, grp in per_event.groupby("month"):
        n_kept = int(len(grp))
        if n_kept < BREADTH_MIN:
            pre_rows.append({
                "month": pd.Timestamp(month), "n_kept": n_kept,
                "preC_ret": 0.0, "dropped": True,
            })
            continue
        sev = grp["severity"].to_numpy().astype(float)
        N = len(sev)
        cap_t = max(NAME_CAP_BASE, NAME_CAP_INV_N / N)
        w, _ = cap_weights(sev, cap_t)
        short_ret = float((w * grp["recipient_ret"].to_numpy()).sum())
        long_ret = float(grp["control_ret"].mean())
        pre_rows.append({
            "month": pd.Timestamp(month), "n_kept": n_kept,
            "preC_ret": short_ret - long_ret, "dropped": False,
        })
    if not pre_rows:
        return pd.DataFrame()
    pre = pd.DataFrame(pre_rows).sort_values("month").reset_index(drop=True)
    leverages = []
    for i in range(len(pre)):
        if pre.at[i, "dropped"]:
            leverages.append(0.0); continue
        if i < VOL_LOOKBACK:
            leverages.append(1.0); continue
        window = pre["preC_ret"].iloc[i - VOL_LOOKBACK:i].to_numpy()
        sigma = float(np.std(window, ddof=1)) if len(window) >= 2 else 0.0
        if sigma <= 0 or not np.isfinite(sigma):
            leverages.append(1.0)
        else:
            leverages.append(float(np.clip(VOL_TARGET_MONTHLY / sigma, 0.0, LEV_CAP)))
    pre["leverage"] = leverages
    pre["ls_return"] = pre["leverage"].to_numpy() * pre["preC_ret"].to_numpy()
    return pre[["month", "ls_return", "n_kept"]].rename(columns={"n_kept": "n_events"})


def _stratum_factor(per_event: pd.DataFrame, mask: pd.Series, label: str,
                     mode: str, french: pd.DataFrame) -> dict:
    """Build the per-stratum monthly factor and run NW alpha."""
    sub = per_event[mask].dropna(subset=["recipient_ret", "control_ret"]).copy()
    if len(sub) < MIN_N_EVENTS:
        return {"label": label, "n_events": int(len(sub)), "skipped": True}

    if mode == "matched":
        fac = _aggregate_matched_monthly(sub)
    elif mode == "rm":
        fac = _aggregate_rm_monthly(sub)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if len(fac) < MIN_N_MONTHS:
        return {"label": label, "n_events": int(len(sub)),
                "n_months": int(len(fac)), "skipped": True}

    panel = fac.merge(
        french[["month", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]],
        on="month", how="inner",
    )
    if len(panel) < MIN_N_MONTHS:
        return {"label": label, "n_events": int(len(sub)),
                "n_months": int(len(panel)), "skipped": True}
    y = panel["ls_return"].to_numpy()
    X = sm.add_constant(panel[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]].to_numpy())
    res = _newey_west_alpha(y, X)
    sharpe = float(y.mean() / y.std() * np.sqrt(12)) if y.std() > 0 else float("nan")
    return {
        "label": label, "n_events": int(len(sub)),
        "n_months": int(len(panel)),
        "sharpe_annual": sharpe,
        "alpha_monthly": res["alpha_monthly"],
        "alpha_annual": res["alpha_annual"],
        "t_alpha": res["t_alpha"], "p_alpha": res["p_alpha"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["matched", "rm"], required=True)
    args = ap.parse_args()

    print(f"[robust-ext] data dir = {DATA}")
    print(f"[robust-ext] mode    = {args.input}")
    print(f"[robust-ext] headline signal spec = {HEADLINE_SPEC['signal_id']}")

    # --- load all inputs once ---
    ev = pd.read_parquet(_d6.EVENTS)
    pairs = pd.DataFrame(
        [json.loads(l) for l in _d6.PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    )
    rets = pd.read_parquet(_d6.RETURNS)
    rets["date"] = pd.to_datetime(rets["date"])
    universe = pd.read_parquet(_d6.UNIVERSE)
    letters = _load_letter_dates(_d6.PAIRS)
    french = pd.read_parquet(DATA / "french_factors_monthly.parquet")
    french["month"] = pd.to_datetime(french["date"]) + pd.offsets.MonthEnd(0)

    print(f"[robust-ext] events={len(ev)}, pairs={len(pairs)}")

    # Build quintiles from monthly returns and merge into events.
    rets_q = _build_size_quintiles(rets)
    rets_q = _build_liquidity_quintiles(rets_q)
    ev["a_start_month"] = pd.to_datetime(ev["a_start_month"])
    ticker_q = (rets_q[["ticker", "date", "size_quintile", "liq_quintile"]]
                .rename(columns={"date": "month"}))
    ev = ev.merge(
        ticker_q, left_on=["ticker", "a_start_month"],
        right_on=["ticker", "month"], how="left",
    )

    # Build the per-event matched panel ONCE for the headline signal.
    size_proxy = _build_size_proxy(rets)
    log_pivot = _build_log_ret_pivot(rets)
    mkt_log = _load_french_market()
    ticker_to_cik = dict(zip(universe["ticker"], universe["cik"]))
    sector_to_tickers: dict[str, list[str]] = (
        universe.groupby("sector")["ticker"].apply(list).to_dict()
    )
    all_months = size_proxy.index
    excl = _build_letter_recipients_per_month(letters, all_months)

    print("[robust-ext] building per-event matched panel for headline signal...")
    per_event, drops = build_per_event_rows(
        HEADLINE_SPEC, ev, log_pivot, mkt_log, size_proxy,
        sector_to_tickers, ticker_to_cik, excl,
    )
    print(f"[robust-ext] per-event panel: rows={len(per_event)}, drops={drops}")

    # The per-event panel keeps cik/sector/ticker but NOT size_q / liq_q.
    # Map them in by joining on (ticker, month).
    pe = per_event.merge(
        ticker_q.rename(columns={"month": "_qmonth"}),
        left_on=["ticker", "month"], right_on=["ticker", "_qmonth"], how="left",
    )

    out: dict = {
        "mode": args.input,
        "baseline": ("Day 6 matched control" if args.input == "matched"
                     else "Day 7 risk-managed (N=4) overlay"),
        "headline_signal_spec": HEADLINE_SPEC["signal_id"],
        "config": {
            "ret_col": HEADLINE_RET_LABEL,
            "min_n_events": MIN_N_EVENTS,
            "min_n_months": MIN_N_MONTHS,
            "nw_lag": NW_LAG,
            "breadth_min": BREADTH_MIN if args.input == "rm" else None,
            "name_cap_base": NAME_CAP_BASE if args.input == "rm" else None,
            "vol_target_annual": (VOL_TARGET_MONTHLY * np.sqrt(12)) if args.input == "rm" else None,
            "vol_lookback_months": VOL_LOOKBACK if args.input == "rm" else None,
            "lev_cap": LEV_CAP if args.input == "rm" else None,
        },
        "strata": {},
    }

    pe_idx = pe.reset_index(drop=True)
    out["strata"]["FULL"] = _stratum_factor(
        pe_idx, pd.Series(True, index=pe_idx.index), "FULL", args.input, french,
    )
    for sec in pe_idx["sector"].dropna().unique():
        out["strata"][f"sector={sec}"] = _stratum_factor(
            pe_idx, pe_idx["sector"] == sec, f"sector={sec}", args.input, french,
        )
    for q in sorted(pe_idx["size_quintile"].dropna().unique()):
        out["strata"][f"size_q{int(q)}"] = _stratum_factor(
            pe_idx, pe_idx["size_quintile"] == q, f"size_q{int(q)}", args.input, french,
        )
    for q in sorted(pe_idx["liq_quintile"].dropna().unique()):
        out["strata"][f"liq_q{int(q)}"] = _stratum_factor(
            pe_idx, pe_idx["liq_quintile"] == q, f"liq_q{int(q)}", args.input, french,
        )

    out_path = DATA / (
        "day7_robustness_matched_summary.json" if args.input == "matched"
        else "day7_robustness_rm_summary.json"
    )
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[robust-ext] wrote {out_path}")

    # --- print summary table ---
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
