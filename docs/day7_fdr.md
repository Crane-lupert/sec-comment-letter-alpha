# Day 7 — Per-stratum FDR analysis

## Why

CLAUDE.md Day 6-7 Rigor Checklist requires Benjamini-Hochberg FDR correction. Day 4 / Day 6 only deflate the 8 pre-registered cells (DSR via Bailey-López de Prado). A reviewer can still slice the cohort by topic or severity band and report a single significant slice — that is multiple-comparison hunting unless every slice is FDR-controlled. This document records the corrected exercise.

## Method

1. Underlying signal: `B_bhar_2m_matched` (the pre-registered headline — corresp_date event, 60-day BHAR, sector+size matched control). The matched-control build (Day 6) is re-used unchanged via direct import.
2. Stratum grid: 1 ALL + 15 topics + 4 severity bands + 5×4 = 20 topic×band cross-cells = 40 base strata. Each evaluated on FULL / IS_2015_2021 / OOS_2022_2024 windows = 120 (stratum, window) cells.
3. For each cell: filter per-event records, aggregate severity-weighted long-short, regress monthly returns on FF5+UMD with Newey-West HAC (lag=6), record alpha+t+p and annualized Sharpe.
4. Eligibility threshold: ≥15 events AND ≥12 months.
5. Apply Benjamini-Hochberg FDR at α=0.05 across ALL eligible cells jointly.

## Headline numbers

- Total cells tested: **120**
- Eligible (≥15 events & ≥12 months): **43**
- Pass nominal p<0.05: **10**
- Pass BH-FDR at α=0.05: **2**

## Top 10 surviving cells (by |t|)

| stratum | window | n_months | n_events | alpha_annual | t | p_raw | p_BH |
|---|---|---|---|---|---|---|---|
| topic=non_gaap_metrics | FULL | 67 | 231 | 0.3276 | +3.11 | 0.0019 | 0.0411 |
| sev_0.5_0.8 | OOS_2022_2024 | 35 | 439 | 0.3056 | +3.10 | 0.0019 | 0.0411 |

## Interpretation

**2 cell(s) survive BH-FDR at α=0.05.** These are robust to multiple-comparison correction across the 43-cell stratification grid; alpha here is not chance-discovered.

Caveat: surviving cells with overlap (e.g. `topic=X` and `topic=X|sev_band`) are not statistically independent — BH controls false-discovery rate under arbitrary positive dependence (Benjamini-Yekutieli would be even more conservative).

## Files

- `scripts/day7_fdr.py` — this analysis
- `data/day7_fdr_summary.json` — full per-cell table + BH summary
- Re-uses (read-only): `scripts/day6_signal_matched.py`, `data/day4_events.parquet`, `data/french_factors_monthly.parquet`.
