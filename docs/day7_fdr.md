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
- Eligible (≥15 events & ≥12 months): **40**
- Pass nominal p<0.05: **3**
- Pass BH-FDR at α=0.05: **0**

## Top 10 surviving cells (by |t|)

_No cells survive BH-FDR at α=0.05._


As a consolation, top 10 by nominal-p (for diagnostic only):

| stratum | window | n_months | alpha_annual | t | p_raw | p_BH |
|---|---|---|---|---|---|---|
| sev_0.5_0.8 | OOS_2022_2024 | 34 | 0.2973 | +3.00 | 0.0027 | 0.1064 |
| topic=non_gaap_metrics|sev_0.2_0.5 | FULL | 22 | -0.3662 | -2.00 | 0.0455 | 0.5225 |
| sev_0.5_0.8 | FULL | 95 | 0.2409 | +1.98 | 0.0478 | 0.5225 |

## Interpretation

**No per-stratum cell survives BH-FDR at α=0.05.** That is the honest answer: even though some individual slices show nominal p<0.05, the count is in the range expected by chance for 40 simultaneous tests. The Day 4/6 headline alphas are pre-registered (only 8 cells, not 100+); selecting any one topic ex-post would be data-mining.

This is a *negative result on the topic-stratification dimension*, not on the main signal. The pre-registered B_bhar_2m_matched alpha remains the headline; topic-level decomposition adds explanatory color but no additional tradeable strata.

## Files

- `scripts/day7_fdr.py` — this analysis
- `data/day7_fdr_summary.json` — full per-cell table + BH summary
- Re-uses (read-only): `scripts/day6_signal_matched.py`, `data/day4_events.parquet`, `data/french_factors_monthly.parquet`.
