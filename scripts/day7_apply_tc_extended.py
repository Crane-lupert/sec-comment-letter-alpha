"""Day 7 transaction cost sensitivity on matched + risk-managed factors.

This is a follow-up to `scripts/day6_apply_tc.py` (which uses the Day 4
sector-mean factor, by pre-registered spec). The pre-registered Day 6 TC
summary is preserved unchanged. This script ADDS two new TC summaries on
the headline-consistent baselines:

  --input matched   : data/day6_factor_returns_matched.parquet   (Day 6 matched control)
  --input rm        : data/day7_risk_managed_factor_returns.parquet (Day 7 N=4 RM overlay)

Output schema is identical to data/day6_post_tc_summary.json so the
dashboard can reuse the same rendering loop.

Outputs:
  data/day7_post_tc_matched_summary.json
  data/day7_post_tc_rm_summary.json

Usage:
    .venv/Scripts/python.exe scripts/day7_apply_tc_extended.py --input matched
    .venv/Scripts/python.exe scripts/day7_apply_tc_extended.py --input rm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

# Reuse helpers from day6_apply_tc verbatim via direct import.
from day6_apply_tc import (  # noqa: E402
    NW_LAG,
    TC_BPS_CONSERVATIVE,
    TC_BPS_REASONABLE,
    TC_BPS_OPTIMISTIC,
    _newey_west_alpha,
    _sharpe_annual,
)
import statsmodels.api as sm  # noqa: E402

REPO_ROOT_LOCAL = _HERE.parents[1]
REPO_ROOT_MAIN = Path("D:/vscode/sec-comment-letter-alpha")


def _resolve(rel: str) -> Path:
    """Prefer the worktree-local path; fall back to the main repo data dir."""
    p_local = REPO_ROOT_LOCAL / rel
    if p_local.exists():
        return p_local
    p_main = REPO_ROOT_MAIN / rel
    return p_main


INPUTS = {
    "matched": {
        "rets": "data/day6_factor_returns_matched.parquet",
        "out": "data/day7_post_tc_matched_summary.json",
        "label": "Day 6 matched control",
    },
    "rm": {
        "rets": "data/day7_risk_managed_factor_returns.parquet",
        "out": "data/day7_post_tc_rm_summary.json",
        "label": "Day 7 risk-managed (N=4) overlay",
    },
}

FRENCH_REL = "data/french_factors_monthly.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=list(INPUTS.keys()), required=True)
    args = ap.parse_args()

    cfg = INPUTS[args.input]
    rets_path = _resolve(cfg["rets"])
    french_path = _resolve(FRENCH_REL)
    # Always write outputs into the main-repo data dir if the worktree
    # doesn't have its own data/ tree (matches existing convention).
    out_root = REPO_ROOT_LOCAL if (REPO_ROOT_LOCAL / "data").exists() else REPO_ROOT_MAIN
    out_path = out_root / cfg["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[tc-ext] input  : {rets_path}  ({cfg['label']})")
    print(f"[tc-ext] output : {out_path}")

    rets = pd.read_parquet(rets_path)
    rets["month"] = pd.to_datetime(rets["month"])
    french = pd.read_parquet(french_path)
    french["month"] = pd.to_datetime(french["date"]) + pd.offsets.MonthEnd(0)

    fac_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    panel = french[["month"] + fac_cols + ["RF"]]

    out = {
        "baseline": cfg["label"],
        "input_parquet": str(rets_path).replace("\\", "/"),
        "signals": {},
        "tc_levels_bps_per_month": {
            "optimistic_5bp": TC_BPS_OPTIMISTIC * 10000,
            "reasonable_10bp": TC_BPS_REASONABLE * 10000,
            "conservative_20bp": TC_BPS_CONSERVATIVE * 10000,
        },
    }

    for sig in sorted(rets["signal_id"].unique()):
        df = rets[rets["signal_id"] == sig].merge(panel, on="month", how="inner")
        df = df.sort_values("month").reset_index(drop=True)
        sig_out = {}
        for label, tc in [
            ("raw", 0.0),
            ("optimistic_5bp", TC_BPS_OPTIMISTIC),
            ("reasonable_10bp", TC_BPS_REASONABLE),
            ("conservative_20bp", TC_BPS_CONSERVATIVE),
        ]:
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

        # Break-even TC (the bps that zero out raw mean monthly return)
        d_full = df.reset_index(drop=True)
        if len(d_full) >= 12:
            mean_m = float(d_full["raw_return"].mean())
            sig_out["break_even_tc_bps"] = mean_m * 10000

        out["signals"][sig] = sig_out

    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[tc-ext] wrote {out_path}\n")

    # Print headline tables — A/B BHAR-2m only.
    headline_sigs = [s for s in out["signals"] if "bhar_2m" in s]
    headline_sigs.sort()
    print(f"{'signal':>26} | {'window':<14} | {'TC':<18} | {'Sharpe':>6} | {'alpha %/y':>10} | {'t':>6} | {'p':>5}")
    print("-" * 100)
    for sig in headline_sigs:
        for w in ["FULL", "IS_2015_2021", "OOS_2022_2024"]:
            for tc_label in ["raw", "optimistic_5bp", "reasonable_10bp", "conservative_20bp"]:
                key = f"{w}_{tc_label}"
                b = out["signals"][sig].get(key)
                if not b:
                    continue
                a = b["alpha_post_tc"]
                print(f"{sig:>26} | {w:<14} | {tc_label:<18} | "
                      f"{b['sharpe_annual_post_tc']:>+5.2f} | "
                      f"{a['alpha_annual']*100:>+8.2f}% | "
                      f"{a['t_alpha']:>+5.2f} | "
                      f"{a['p_alpha']:>5.3f}")
            print()

        be = out["signals"][sig].get("break_even_tc_bps", float("nan"))
        print(f"  -> {sig} break-even TC: {be:.1f} bps/month before alpha = 0\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
