# Pre-registration: Day 4 cross-section event-study spec

**Status**: Locked at the git commit containing this file. Spec choices below
are frozen; subsequent code runs may not pick a different window/horizon/measure
without an explicit amendment commit citing this file.

**Date**: 2026-04-26 KST

## Why pre-register Day 4

CLAUDE.md mentions both 60- and 90-day post-letter abnormal returns, both CAR
and BHAR are common in event studies, and three baseline factor models. Without
locking the spec, picking the best of {60d, 90d} × {CAR, BHAR} × {FF3, FF5,
FF5+UMD} after seeing results is a 12-cell multiple-comparison hunt, and the
final reported Sharpe would be inflated.

## Locked spec

### Forward-return measure
- **Main**: BHAR (buy-and-hold abnormal return) — academic standard for event
  studies > 1 month, less sensitive to compounding artifacts than CAR for
  multi-month windows.
- **Robustness reported alongside**: CAR over the same window.

### Horizons
- **Main**: t+1 to t+60 calendar days (≈ 1 quarter, the SEC review cadence
  matches; lag p90=28d so most CORRESP arrive within window).
- **Robustness reported alongside**: t+1 to t+90, t+1 to t+30.

### Baseline factor model for orthogonalization
- **Main**: FF5 + Carhart UMD (= 6-factor; Fama-French 5 + momentum).
- **Robustness reported alongside**: FF3, FF3+UMD.
- Source: Kenneth French data library, monthly factors, US.

### Signal definitions (two distinct tradeable signals)

The hypothesis ("comment letter content predicts post-letter returns") is
about a tradeable signal at letter date. Two flavors are pre-registered:

#### Signal A — UPLOAD-only (early-tradeable)
- **Event date**: `upload_date`
- **Information set at event**: UPLOAD letter text alone (SEC's outbound
  comments). CORRESP has not been filed yet.
- **Features used**: UPLOAD `severity_mean`, UPLOAD `topics_consensus`, UPLOAD
  `resolution_signal` (4-class: accepted/partial/ongoing/unknown).
- **Forward-return window**: upload_date+1 to upload_date+60.

#### Signal B — UPLOAD + CORRESP (late-tradeable)
- **Event date**: `corresp_date` (= upload_date + response_lag_days).
- **Information set at event**: BOTH letters available.
- **Features used**: all 12-dim feature vector from `data/day4_pairs.jsonl`.
- **Forward-return window**: corresp_date+1 to corresp_date+60.

Signal B uses a strictly later event date (median +12d) and a strictly later
return window. Both signals' returns are computed without leakage.

### Portfolio construction
- **Long-short**: severity-weighted short on letter recipients (high severity
  = larger short), control matched-by-size-and-sector long on non-letter R3K
  firms within ±20% market cap and same GICS sector. Equal-weight within
  long basket.
- **Rebalance**: monthly (calendar month-end).
- **Sector neutralization**: each month's portfolio is sector-residualized
  (subtract sector-mean exposure).
- **Net exposure**: dollar-neutral (long $ = short $).

### Test statistics reported
- Mean monthly return (raw + post-FF5+UMD residual).
- Annualized Sharpe (raw + residual).
- t-statistic with Newey-West HAC SE, lag = 6 months.
- 95% CI via month-clustered bootstrap (B = 1000 resamples).
- Deflated Sharpe Ratio (Bailey-López de Prado) using n_trials = 12 (the
  pre-registered cells: 2 signals × 2 measures × 3 horizons).

### IS / OOS split
- **IS**: 2015-01 through 2021-12 (84 months).
- **OOS**: 2022-01 through 2024-12 (36 months) — frozen, not touched until
  Day 6-7 rigor pass complete.
- All hyperparameter choices (severity weighting, sector match window) made
  on IS only. OOS Sharpe is the headline result.

### What this pre-registration does NOT permit
- Reporting only the best window/measure/baseline cell.
- Re-defining the long/short matching rule after seeing returns.
- Touching OOS data before the freeze release date.
- Excluding sectors/firms post-hoc to inflate Sharpe.
- Using CORRESP-derived features at upload_date (would be Signal-A leakage).

## Sample size acknowledgement

Pre-registration is binding regardless of the sample-size decision. As of this
commit, 867 R3K UPLOAD-CORRESP pairs are extracted (603 in 2015-2024). A
parallel extraction is expanding to ~5000 records (2015-2024 priority). Final
sample size will be reported in Day 4 results without changing the spec.
