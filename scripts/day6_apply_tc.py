"""Day 6 transaction cost model — post-cost Sharpe + alpha.

CLAUDE.md spec: 10 bps fixed + 0.05 x ADV participation. We implement the
fixed-bps part precisely; ADV participation requires per-firm daily volume
which we don't have for the full panel. The bps cost is a conservative
upper bound for low-ADV impact.

Cost model (per-month deduction from long-short return):
  optimistic_5bp:    5 bps/month -- high persistence + tight execution
  reasonable_10bp:  10 bps/month -- 50% turnover + commission + spread
  conservative_20bp: 20 bps/month -- 100% turnover, both legs at 10 bps

Outputs: data/day6_post_tc_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RETS = REPO_ROOT / "data" / "day4_factor_returns.parquet"
FRENCH = REPO_ROOT / "data" / "french_factors_monthly.parquet"
OUT = REPO_ROOT / "data" / "day6_post_tc_summary.json"

NW_LAG = 6
TC_BPS_CONSERVATIVE = 0.0020
TC_BPS_REASONABLE = 0.0010
TC_BPS_OPTIMISTIC = 0.0005


def _newey_west_alpha(y: np.ndarray, X: np.ndarray, lag: int = NW_LAG) -> dict:
    if len(y) < 12:
        return {"alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "t_alpha": float("nan"), "p_alpha": float("nan"), "se_monthly": float("nan")}
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    a = float(m.params[0])
    se = float(m.bse[0])
    return {
        "alpha_monthly": a, "alpha_annual": a * 12,
        "se_monthly": se, "t_alpha": a / se if se else float("nan"),
        "p_alpha": float(m.pvalues[0]),
    }


def _sharpe_annual(returns: np.ndarray) -> float:
    if len(returns) < 12 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(12))


def main() -> int:
    rets = pd.read_parquet(RETS)
    rets["month"] = pd.to_datetime(rets["month"])
    french = pd.read_parquet(FRENCH)
    french["month"] = pd.to_datetime(french["date"]) + pd.offsets.MonthEnd(0)

    fac_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    panel = french[["month"] + fac_cols + ["RF"]]

    out = {"signals": {}, "tc_levels_bps_per_month": {
        "optimistic_5bp": TC_BPS_OPTIMISTIC * 10000,
        "reasonable_10bp": TC_BPS_REASONABLE * 10000,
        "conservative_20bp": TC_BPS_CONSERVATIVE * 10000,
    }}

    for sig in sorted(rets["signal_id"].unique()):
        df = rets[rets["signal_id"] == sig].merge(panel, on="month", how="inner")
        df = df.sort_values("month").reset_index(drop=True)
        sig_out = {}
        for label, tc in [("raw", 0.0),
                          ("optimistic_5bp", TC_BPS_OPTIMISTIC),
                          ("reasonable_10bp", TC_BPS_REASONABLE),
                          ("conservative_20bp", TC_BPS_CONSERVATIVE)]:
            for window_label, mask in [
                ("FULL", pd.Series(True, index=df.index)),
                ("IS_2015_2021", (df["month"] >= "2015-01-01") & (df["month"] <= "2021-12-31")),
                ("OOS_2022_2024", (df["month"] >= "2022-01-01") & (df["month"] <= "2024-12-31")),
            ]:
                d = df[mask].reset_index(drop=True)
                y = d["raw_return"].values - tc
                X = sm.add_constant(d[fac_cols].values)
                alpha = _newey_west_alpha(y, X, lag=NW_LAG)
                sharpe = _sharpe_annual(y)
                key = f"{window_label}_{label}"
                sig_out[key] = {
                    "n_months": int(len(d)),
                    "tc_bps_per_month": tc * 10000,
                    "mean_monthly_post_tc": float(y.mean()) if len(y) else float("nan"),
                    "sharpe_annual_post_tc": sharpe,
                    "alpha_post_tc": alpha,
                }
        out["signals"][sig] = sig_out

        # Break-even TC (the bps that zero out raw mean monthly return)
        d_full = df.reset_index(drop=True)
        if len(d_full) >= 12:
            mean_m = float(d_full["raw_return"].mean())
            sig_out["break_even_tc_bps"] = mean_m * 10000

    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[tc] wrote {OUT}\n")

    print(f"{'signal':>14} | {'window':<14} | {'TC':<18} | {'Sharpe':>6} | {'alpha %/y':>10} | {'t':>6}")
    print("-" * 90)
    for sig in ["A_bhar_2m", "B_bhar_2m"]:
        for w in ["FULL", "IS_2015_2021", "OOS_2022_2024"]:
            for tc_label in ["raw", "optimistic_5bp", "reasonable_10bp", "conservative_20bp"]:
                key = f"{w}_{tc_label}"
                b = out["signals"][sig][key]
                print(f"{sig:>14} | {w:<14} | {tc_label:<18} | "
                      f"{b['sharpe_annual_post_tc']:>+5.2f} | "
                      f"{b['alpha_post_tc']['alpha_annual']*100:>+8.2f}% | "
                      f"{b['alpha_post_tc']['t_alpha']:>+5.2f}")
            print()

        be = out["signals"][sig].get("break_even_tc_bps", float("nan"))
        print(f"  -> {sig} break-even TC: {be:.1f} bps/month before alpha = 0\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
