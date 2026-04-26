"""Day 6 — print head-to-head comparison: matched vs sector-mean control.

Reads:
  - data/day4_alpha_summary.json          (sector-mean residualization, locked)
  - data/day6_alpha_summary_matched.json  (sector + size matched)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY4 = REPO_ROOT / "data" / "day4_alpha_summary.json"
DAY6 = REPO_ROOT / "data" / "day6_alpha_summary_matched.json"


def main() -> int:
    d4 = json.loads(DAY4.read_text(encoding="utf-8"))
    d6 = json.loads(DAY6.read_text(encoding="utf-8"))

    # Map matched sig -> day4 sig (drop "_matched" suffix)
    matched_sigs = sorted(d6["signals"].keys())
    rows = []
    print("=== sector-matched vs sector-mean-residualized comparison ===")
    print(f"  {'signal':<14} | {'window':<6} | {'Sharpe (matched) -> (sector-mean)':<37} | "
          f"{'alpha_t (matched) -> (sector-mean)':<35}")
    for sig_m in matched_sigs:
        sig_base = sig_m.replace("_matched", "")
        if sig_base not in d4["signals"]:
            continue
        for window_key, window_label in [("FULL", "FULL"), ("IS_2015_2021", "IS"), ("OOS_2022_2024", "OOS")]:
            m_blk = d6["signals"][sig_m].get(window_key, {})
            b_blk = d4["signals"][sig_base].get(window_key, {})
            if not m_blk or not b_blk:
                continue
            m_sh = m_blk.get("raw_sharpe_annual", float("nan"))
            b_sh = b_blk.get("raw_sharpe_annual", float("nan"))
            m_t = (m_blk.get("alpha", {}) or {}).get("t_alpha", float("nan"))
            b_t = (b_blk.get("alpha", {}) or {}).get("t_alpha", float("nan"))
            print(f"  {sig_base:<14} | {window_label:<6} | "
                  f" {m_sh:>5.2f} -> {b_sh:>5.2f}                          | "
                  f" {m_t:>5.2f} -> {b_t:>5.2f}")
            rows.append({
                "signal": sig_base, "window": window_label,
                "sharpe_matched": m_sh, "sharpe_sector_mean": b_sh,
                "alpha_t_matched": m_t, "alpha_t_sector_mean": b_t,
            })
    return 0


if __name__ == "__main__":
    sys.exit(main())
