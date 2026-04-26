"""Orthogonalize signal returns against FF5 + UMD baseline.

For each signal_id: regress monthly raw_return on Mkt-RF + SMB + HML + RMW +
CMA + UMD (intercept = annualized alpha). Report:
  - alpha (monthly + annualized)
  - Newey-West HAC SE (lag = 6 months)
  - t-statistic
  - month-clustered bootstrap 95% CI (B = 1000)
  - residual Sharpe annualized
  - raw Sharpe annualized
  - DSR (Bailey-Lopez de Prado) using n_trials = 8 (the 8 cells in SIGNAL_SPECS)

Splits results into IS (2015-2021) and OOS (2022-2024) per pre-registration.

Outputs:
  - data/day4_alpha_summary.json  (one block per signal_id, per IS/OOS)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RETS = REPO_ROOT / "data" / "day4_factor_returns.parquet"
FRENCH = REPO_ROOT / "data" / "french_factors_monthly.parquet"
OUT = REPO_ROOT / "data" / "day4_alpha_summary.json"

IS_START = "2015-01-01"
IS_END = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END = "2024-12-31"

N_TRIALS_DSR = 8
NW_LAG = 6
B_BOOTSTRAP = 1000
RNG_SEED = 42


def newey_west_alpha(y: np.ndarray, X: np.ndarray, lag: int = NW_LAG) -> dict:
    if len(y) < 12:
        return {"n": len(y), "alpha_monthly": float("nan"), "se_monthly": float("nan"),
                "t_alpha": float("nan"), "alpha_annual": float("nan"),
                "se_annual": float("nan"), "p_alpha": float("nan"), "betas": {}, "r2": float("nan")}
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    alpha = float(model.params[0]); se = float(model.bse[0])
    t = alpha / se if se else float("nan")
    return {
        "n": int(len(y)), "alpha_monthly": alpha, "alpha_annual": alpha * 12,
        "se_monthly": se, "se_annual": se * np.sqrt(12), "t_alpha": t,
        "p_alpha": float(model.pvalues[0]),
        "betas": {f"b_{i}": float(b) for i, b in enumerate(model.params[1:])},
        "r2": float(model.rsquared),
    }


def cluster_bootstrap_ci(returns: np.ndarray, n_boot: int = B_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < 12: return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(returns[idx].mean()))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def deflated_sharpe(sr_observed: float, n_obs: int, n_trials: int) -> float:
    from scipy.stats import norm
    if n_obs < 12 or n_trials < 1: return float("nan")
    gamma = 0.5772156649
    e_max = (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    sr_var = (1 + 0.5 * sr_observed ** 2) / n_obs
    z = (sr_observed - e_max * np.sqrt(sr_var)) / np.sqrt(sr_var)
    return float(norm.cdf(z))


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 12 or returns.std() == 0: return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(12))


def analyze_window(rets: pd.DataFrame, french: pd.DataFrame, label: str, sig_id: str) -> dict:
    df = rets[rets["signal_id"] == sig_id].copy()
    df["month"] = pd.to_datetime(df["month"])
    fdf = french.copy()
    fdf["month"] = pd.to_datetime(fdf["date"]) + pd.offsets.MonthEnd(0)
    fdf = fdf[["month", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]]
    merged = df.merge(fdf, on="month", how="inner").sort_values("month").reset_index(drop=True)
    if merged.empty:
        return {"label": label, "signal_id": sig_id, "n_months": 0}
    y = merged["raw_return"].values
    X = sm.add_constant(merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]].values)
    nw = newey_west_alpha(y, X, lag=NW_LAG)
    raw_sharpe = annualized_sharpe(y)
    if len(y) >= 12:
        ols = sm.OLS(y, X).fit()
        residuals = y - ols.predict(X)
        residual_sharpe = annualized_sharpe(residuals)
    else:
        residual_sharpe = float("nan")
    ci_lo, ci_hi = cluster_bootstrap_ci(y)
    dsr = deflated_sharpe(raw_sharpe, len(y), N_TRIALS_DSR)
    return {
        "label": label, "signal_id": sig_id, "n_months": int(len(merged)),
        "month_range": f"{merged['month'].min().date()} .. {merged['month'].max().date()}",
        "mean_monthly_raw": float(merged["raw_return"].mean()),
        "std_monthly_raw": float(merged["raw_return"].std()),
        "raw_sharpe_annual": raw_sharpe,
        "residual_sharpe_annual": residual_sharpe,
        "alpha": nw,
        "bootstrap_95ci_mean_monthly": [ci_lo, ci_hi],
        "deflated_sharpe_ratio": dsr,
        "n_trials_for_dsr": N_TRIALS_DSR,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(RETS),
                    help="Path to factor-returns parquet (default: data/day4_factor_returns.parquet)")
    ap.add_argument("--output", type=str, default=str(OUT),
                    help="Path to write alpha summary JSON (default: data/day4_alpha_summary.json)")
    args = ap.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    rets = pd.read_parquet(in_path)
    french = pd.read_parquet(FRENCH)
    sig_ids = sorted(rets["signal_id"].unique())
    print(f"[ortho] input={in_path}")
    print(f"[ortho] signals: {sig_ids}")
    out: dict = {"signals": {}}
    for sig in sig_ids:
        out["signals"][sig] = {
            "FULL": analyze_window(rets, french, "FULL", sig),
            "IS_2015_2021": analyze_window(
                rets[(rets["month"] >= IS_START) & (rets["month"] <= IS_END)], french, "IS", sig),
            "OOS_2022_2024": analyze_window(
                rets[(rets["month"] >= OOS_START) & (rets["month"] <= OOS_END)], french, "OOS", sig),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[ortho] wrote {out_path}")
    print("\n=== Headline (signal | window | n | raw_sharpe | resid_sharpe | alpha_t | dsr) ===")
    for sig, blocks in out["signals"].items():
        for label, b in blocks.items():
            if not b.get("alpha"): continue
            print(f"  {sig:>14} | {label:<14} | n={b['n_months']:>3} | "
                  f"sharpe={b['raw_sharpe_annual']:>5.2f} | resid={b['residual_sharpe_annual']:>5.2f} | "
                  f"alpha_t={b['alpha']['t_alpha']:>5.2f} | dsr={b['deflated_sharpe_ratio']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
