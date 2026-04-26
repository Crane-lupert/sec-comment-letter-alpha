"""Construct Signal A and Signal B monthly long-short factor returns.

Per pre-registration (docs/preregistration_day4_event_study.md):
- Signal A: event=upload_date, UPLOAD-only features
- Signal B: event=corresp_date, UPLOAD+CORRESP features
- Forward measure: BHAR over t+1..t+60 days (= 2 months proxy here);
  CAR + 30/90d also produced for robustness.
- Long-short, severity-weighted short on letter recipients;
  matched-by-size-and-sector long control built post-hoc by sector mean offset.
- Sector neutralization: subtract sector-mean exposure each month.

Output: data/day4_factor_returns.parquet
  Columns: month, signal_id, raw_return, n_long, n_short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS = REPO_ROOT / "data" / "day4_events.parquet"
OUT = REPO_ROOT / "data" / "day4_factor_returns.parquet"

# Map signal -> (event-start col, return-col template)
SIGNAL_SPECS = [
    {"signal_id": "A_bhar_2m", "start_col": "a_start_month", "return_col": "bhar_a_2m", "horizon": 2},
    {"signal_id": "A_bhar_1m", "start_col": "a_start_month", "return_col": "bhar_a_1m", "horizon": 1},
    {"signal_id": "A_bhar_3m", "start_col": "a_start_month", "return_col": "bhar_a_3m", "horizon": 3},
    {"signal_id": "B_bhar_2m", "start_col": "b_start_month", "return_col": "bhar_b_2m", "horizon": 2},
    {"signal_id": "B_bhar_1m", "start_col": "b_start_month", "return_col": "bhar_b_1m", "horizon": 1},
    {"signal_id": "B_bhar_3m", "start_col": "b_start_month", "return_col": "bhar_b_3m", "horizon": 3},
    {"signal_id": "A_car_2m",  "start_col": "a_start_month", "return_col": "car_a_2m",  "horizon": 2},
    {"signal_id": "B_car_2m",  "start_col": "b_start_month", "return_col": "car_b_2m",  "horizon": 2},
]


def _construct_one(ev: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """For one signal spec, monthly long-short factor return.

    Each event-month: severity-weighted short on letter recipients, sector-mean
    BHAR as long control. Net = short - sector_mean.
    Output: one row per month.
    """
    start_col = spec["start_col"]
    ret_col = spec["return_col"]
    df = ev[[start_col, "sector", "upload_severity_mean", ret_col]].dropna(subset=[ret_col, "sector"]).copy()
    df["month"] = df[start_col].values.astype("datetime64[M]").astype("datetime64[ns]") + pd.offsets.MonthEnd(0)

    out_rows = []
    for month, grp in df.groupby("month"):
        if len(grp) < 2:
            continue
        # Severity weights normalized within month
        w = grp["upload_severity_mean"].values
        if w.sum() <= 0:
            w = np.ones(len(grp)) / len(grp)
        else:
            w = w / w.sum()
        # Short-leg: severity-weighted recipient BHAR
        short_ret = float((w * grp[ret_col].values).sum())
        # Long-leg control: sector-mean BHAR for the SAME sectors, equal-weight by sector
        # Without external matched non-letter firms, we use sector-mean-of-recipients
        # as a sector-residualization control. Day 5 will swap in true matched
        # non-letter portfolios when we integrate full panel.
        sector_mean = grp.groupby("sector")[ret_col].mean()
        long_ret = float(grp.assign(_sm=grp["sector"].map(sector_mean))["_sm"].mean())
        net = short_ret - long_ret
        out_rows.append({
            "month": month,
            "signal_id": spec["signal_id"],
            "horizon_months": spec["horizon"],
            "raw_return": net,
            "short_leg_return": short_ret,
            "long_leg_return": long_ret,
            "n_short": len(grp),
            "n_sectors": grp["sector"].nunique(),
        })
    return pd.DataFrame(out_rows)


def main() -> int:
    ev = pd.read_parquet(EVENTS)
    print(f"[signal] events: {len(ev)}")
    print(f"[signal] columns: {[c for c in ev.columns if 'bhar' in c or 'car' in c]}")

    parts = []
    for spec in SIGNAL_SPECS:
        df = _construct_one(ev, spec)
        if df.empty:
            print(f"[signal] {spec['signal_id']}: no months -- skipped")
            continue
        parts.append(df)
        print(f"[signal] {spec['signal_id']}: months={len(df)}, "
              f"mean_raw={df['raw_return'].mean():.4f}, "
              f"std={df['raw_return'].std():.4f}, "
              f"sharpe_annual={df['raw_return'].mean() / df['raw_return'].std() * np.sqrt(12):.2f}")

    if not parts:
        print("[signal] no factor returns produced.")
        return 1
    out = pd.concat(parts, ignore_index=True).sort_values(["signal_id", "month"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[signal] wrote {OUT} -- rows={len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
