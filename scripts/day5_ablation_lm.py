"""Day 5 ablation: does Signal B's alpha survive FF5+UMD+LM orthogonalization?

For pre-registered main cell B_bhar_2m, regress monthly returns against:
  baseline: Mkt-RF + SMB + HML + RMW + CMA + UMD  (Day 4)
  baseline + LM: above + LM_neg_ratio_factor (Day 5 added)

If residual alpha shrinks substantially when LM is added, Signal B was
LM-correlated. If it stays stable, Signal B carries info LM does not.

Outputs to console + data/day5_ablation_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_RETS = REPO_ROOT / "data" / "day4_factor_returns.parquet"
FRENCH = REPO_ROOT / "data" / "french_factors_monthly.parquet"
LM = REPO_ROOT / "data" / "day5_lm_factor.parquet"
OUT = REPO_ROOT / "data" / "day5_ablation_summary.json"

NW_LAG = 6


def _orthogonalize(y: np.ndarray, X: np.ndarray) -> dict:
    if len(y) < 12:
        return {"n": int(len(y)), "alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "t_alpha": float("nan"), "p_alpha": float("nan"), "r2": float("nan"),
                "se_monthly": float("nan")}
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    alpha = float(m.params[0])
    se = float(m.bse[0])
    return {
        "n": int(len(y)),
        "alpha_monthly": alpha,
        "alpha_annual": alpha * 12,
        "se_monthly": se,
        "t_alpha": alpha / se if se else float("nan"),
        "p_alpha": float(m.pvalues[0]),
        "r2": float(m.rsquared),
    }


def main() -> int:
    rets = pd.read_parquet(SIGNAL_RETS)
    rets["month"] = pd.to_datetime(rets["month"])
    french = pd.read_parquet(FRENCH)
    french["month"] = pd.to_datetime(french["date"]) + pd.offsets.MonthEnd(0)
    lm = pd.read_parquet(LM)
    lm["month"] = pd.to_datetime(lm["month"])
    lm = lm[["month", "ls_return"]].rename(columns={"ls_return": "LM_ls"})

    panel = french.merge(lm, on="month", how="left").fillna(0.0)
    fac_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]

    out = {"signals": {}}
    for sig in ["A_bhar_2m", "B_bhar_2m"]:
        df = rets[rets["signal_id"] == sig].merge(panel, on="month", how="inner")
        df = df.sort_values("month").reset_index(drop=True)
        y = df["raw_return"].values

        # Baseline: FF5+UMD
        X_base = sm.add_constant(df[fac_cols].values)
        base = _orthogonalize(y, X_base)

        # Extended: + LM
        X_lm = sm.add_constant(df[fac_cols + ["LM_ls"]].values)
        lm_res = _orthogonalize(y, X_lm)

        # Quick split: IS / OOS
        m_is = (df["month"] >= "2015-01-01") & (df["month"] <= "2021-12-31")
        m_oos = (df["month"] >= "2022-01-01") & (df["month"] <= "2024-12-31")

        out["signals"][sig] = {
            "FULL": {
                "baseline_FF5_UMD": base,
                "extended_FF5_UMD_LM": lm_res,
                "alpha_change_pp_annual": (lm_res["alpha_annual"] - base["alpha_annual"]) * 100,
            },
            "IS_2015_2021": {
                "baseline_FF5_UMD": _orthogonalize(y[m_is], X_base[m_is]),
                "extended_FF5_UMD_LM": _orthogonalize(y[m_is], X_lm[m_is]),
            },
            "OOS_2022_2024": {
                "baseline_FF5_UMD": _orthogonalize(y[m_oos], X_base[m_oos]),
                "extended_FF5_UMD_LM": _orthogonalize(y[m_oos], X_lm[m_oos]),
            },
        }

    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[ablation] wrote {OUT}\n")

    print(f"{'signal':>14} | {'window':<14} | {'base alpha':>10} | {'+LM alpha':>10} | {'Δ pp/yr':>8} | {'base_t':>6} | {'+LM_t':>6}")
    print("-" * 95)
    for sig, blocks in out["signals"].items():
        for w, b in blocks.items():
            base = b["baseline_FF5_UMD"]
            ext = b["extended_FF5_UMD_LM"]
            d_alpha = (ext["alpha_annual"] - base["alpha_annual"]) * 100
            print(f"{sig:>14} | {w:<14} | {base['alpha_annual']*100:>+8.2f}%/y | "
                  f"{ext['alpha_annual']*100:>+8.2f}%/y | {d_alpha:>+6.2f}pp | "
                  f"{base['t_alpha']:>5.2f} | {ext['t_alpha']:>5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
