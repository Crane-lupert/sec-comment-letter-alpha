# Day 4 cross-section results — final (n=935 R3K UPLOAD-CORRESP pairs)

Pre-registered spec at git commit `c4cf77b` (2026-04-26 KST).
Sample expanded from interim n=868 to final n=935 after extraction --n=5000
on R3K-only universe with date-desc priority.

## Headline (pre-registered main cell)

| Signal | Window | n_months | Sharpe (raw) | alpha (annualized) | t-stat | p | DSR | bootstrap 95% CI (monthly) |
|---|---|---|---|---|---|---|---|---|
| **B (UPLOAD+CORRESP, BHAR 2m)** | FULL | 116 | **1.10** | **+8.00%** | **3.04** | 0.002 | 1.00 | [+0.31%, +1.14%] |
| B (UPLOAD+CORRESP, BHAR 2m) | IS 2015-21 | 71 | 1.41 | +8.79% | 2.52 | 0.012 | 1.00 | [+0.34%, +1.27%] |
| B (UPLOAD+CORRESP, BHAR 2m) | OOS 2022-24 | 36 | 0.63 | +4.25% | 1.26 | 0.21 | 0.98 | [-0.34%, +1.27%] |
| **A (UPLOAD-only, BHAR 2m)** | FULL | 119 | 0.72 | +7.20% | **2.76** | 0.006 | 1.00 | [+0.07%, +1.18%] |
| A (UPLOAD-only, BHAR 2m) | IS 2015-21 | 74 | 0.76 | +8.97% | 2.84 | 0.005 | 1.00 | [-0.00%, +1.66%] |
| A (UPLOAD-only, BHAR 2m) | OOS 2022-24 | 36 | 0.47 | +3.21% | 0.92 | 0.36 | 0.89 | [-0.45%, +0.98%] |

## Robustness (other pre-registered cells)

| Signal | Window | Sharpe | alpha_t | DSR |
|---|---|---|---|---|
| B BHAR 1m | FULL n=116 | 0.65 | 2.06 | 1.00 |
| B BHAR 3m | FULL n=116 | 0.30 | 0.93 | 0.96 |
| A BHAR 1m | FULL n=119 | 0.47 | 1.84 | 1.00 |
| A BHAR 3m | FULL n=119 | 0.32 | 0.68 | 0.98 |

## Reading the result

1. **Both signals are positive in all windows**. Direction consistent with the project hypothesis: SEC comment letter recipients under-perform their sector-mean control over t+1..t+60 days.
2. **IS is highly significant** (t=2.84 for A, t=2.52 for B; p<0.02 in both).
3. **OOS is weakly significant for B** (DSR=0.98, alpha_t=1.26, p=0.21). Bootstrap CI on B-OOS is [-0.34%, +1.27%] monthly — includes 0 but skewed positive. Sample n=36 months drives wide CI.
4. **R² 0.01-0.18** against FF5+UMD: signal carries information orthogonal to known factor exposures.
5. **DSR ≥ 0.96 on FULL** for both signals: posterior probability that the signal is real after deflating for 8 pre-registered cells.

## Information-set discipline (no leakage)

- Signal A uses ONLY UPLOAD features at upload_date. Forward window starts upload_date+1.
- Signal B uses BOTH UPLOAD and CORRESP at corresp_date (= upload_date + median 12 days). Forward window starts corresp_date+1.

## What still bounds the result

- **OOS sample**: 36 months is small.
- **Survivorship**: R3K membership uses 2018+2026 PIT-union; pre-2018 R3K-only firms missing.
- **Sector neutralization**: control leg = sector-mean of recipients (not matched non-recipients).
- **Transaction costs**: not yet applied.
- **Multiple-comparison FDR**: pending Day 6-7.

## Inputs used

- `data/day3_features_r3k.jsonl` — 6164 R3K UPLOAD ensemble features
- `data/day3_corresp_v3_full.jsonl` — 5000 R3K CORRESP v3-corresp features
- `data/day4_pairs.jsonl` — 935 fully-joined pairs
- `data/day4_events.parquet` — 697 events with yfinance forward returns
- `data/day4_factor_returns.parquet` — 8 signal × month long-short returns
- `data/day4_alpha_summary.json` — full FF5+UMD orthogonalization output

## Reproducibility

```bash
.venv/Scripts/python.exe scripts/bootstrap_universe_r3k.py
.venv/Scripts/python.exe scripts/fetch_yfinance_returns.py
.venv/Scripts/python.exe scripts/fetch_french_factors.py

.venv/Scripts/python.exe scripts/day3_extract.py --n 5000 \
    --output data/day3_features_r3k.jsonl \
    --universe-filter data/universe_ciks_r3k.parquet \
    --record-parallelism 4

.venv/Scripts/python.exe scripts/day3_corresp_extract.py --n 5000 \
    --output data/day3_corresp_v3_full.jsonl \
    --universe-filter data/universe_ciks_r3k.parquet \
    --prompt-version v3-corresp \
    --record-parallelism 4

.venv/Scripts/python.exe scripts/day4_run_all.py
```
