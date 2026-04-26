# Day 6 — Sector + size matched non-letter long control

## Why

The Day 4 spec (`docs/preregistration_day4_event_study.md`) calls for a
*matched-by-size-and-sector long on non-letter firms within ±20% market cap and
same GICS sector*. The Day 4 implementation (`scripts/day4_construct_signal.py`,
locked at commit 4efaa22) instead uses the **sector-mean BHAR of letter
recipients** as the long leg — i.e. sector residualization, not a real matched
control. Day 6 builds the proper matched control and re-runs the orthogonalization
to see how much of the Day 4 alpha survives.

## Method

1. **Per-month clean-control universe.** For each calendar month M, the
   "letter-receiving" set is every CIK with an `upload_date` in
   `[M − 12 months, M + 3 months]`. Any firm in that set is excluded from the
   long-leg pool for month M (12-month look-back to drop recent recipients;
   3-month look-ahead to avoid contaminating the control with firms about
   to receive a letter).
2. **Size proxy.** `data/r3k_monthly_returns.parquet` contains
   `[date, ticker, price, log_ret]` only — no shares-outstanding, no volume.
   We use **adjusted close at the end of the month prior to the event** as the
   size/tradability proxy (point-in-time, no look-ahead). This is a known
   limitation: price-as-size is correlated with share-price *level*, not with
   market cap. A ±20% price band groups firms with similar price levels, which
   is a noisier match than ±20% on market cap. Properly fixing this needs a
   shares-outstanding history (Compustat or yfinance fast info per ticker per
   filing).
3. **Match per event.** For each letter event `(cik, sector, event_month)`:
   pick K=5 same-sector R3K firms (excluding the event firm and all firms in
   the rolling letter-recipient set) whose size proxy is within ±20% of the
   letter firm's. Closest on size first. If <5 match, take all available; if
   0 match, drop the event.
4. **Matched-pair returns.** Equal-weight mean BHAR (or CAR) of the matched
   K-tuple over the same forward window (1m, 2m, 3m), benchmarked against the
   same `Mkt-RF + RF` log-market series Day 4 uses.
5. **Monthly long-short factor.**
   ```
   short_leg(month) = severity-weighted Σ recipient BHAR
   long_leg(month)  = mean of per-event matched-control BHAR
   raw_return       = short_leg − long_leg
   ```
   Same severity normalization (z-sum within month, fall back to equal-weight
   if Σ severity ≤ 0) as Day 4. Months with <2 events skipped.

## Spec

| param | value |
|---|---|
| K (controls per event) | 5 |
| size band | ±20% |
| look-back excluding letter recipients | 12 months |
| look-ahead excluding letter recipients | 3 months |
| size proxy | adjusted-close price at end of prior month |
| signals | A_{bhar,car}_{1,2,3}m_matched, B_{bhar,car}_{1,2,3}m_matched |

## Coverage

Total events in `data/day4_events.parquet`: **697**. Across all 8 signals the
match step retains **680** events (97.6%). Drop reasons:

| reason | count (typical signal) |
|---|---|
| `no_recipient_ret` (event firm has no horizon return) | 0–6 (varies by horizon) |
| `no_match` (no same-sector ±20% non-letter firm) | 7 |
| `no_size_month` (prior month not in panel) | 3–5 |
| `letter_size_nan` (letter firm price missing at proxy month) | 1–5 |

Per-event `n_controls` distribution: median = 5, mean ≈ 4.88, min = 1, max = 5.

## Headline comparison: matched vs sector-mean control

| signal | window | Sharpe (matched) → (sector-mean) | alpha_t (matched) → (sector-mean) |
|---|---|---|---|
| A_bhar_1m | FULL | 0.65 → 0.47 | 1.90 → 1.84 |
| A_bhar_1m | IS | −0.13 → 0.42 | 0.19 → 1.91 |
| A_bhar_1m | OOS | **1.86 → 0.58** | **2.23 → 1.61** |
| A_bhar_2m | FULL | 0.52 → 0.72 | 1.07 → 2.76 |
| A_bhar_2m | IS | 0.02 → 0.76 | −0.18 → 2.84 |
| A_bhar_2m | OOS | **1.37 → 0.47** | **2.87 → 0.92** |
| A_bhar_3m | FULL | 0.50 → 0.32 | 0.82 → 0.68 |
| A_bhar_3m | IS | −0.05 → 0.34 | −0.54 → 0.57 |
| A_bhar_3m | OOS | **1.32 → −0.27** | **1.78 → −1.12** |
| A_car_2m | FULL | 0.52 → 0.72 | 1.07 → 2.76 |
| A_car_2m | IS | 0.02 → 0.76 | −0.18 → 2.84 |
| A_car_2m | OOS | 1.37 → 0.47 | 2.87 → 0.92 |
| B_bhar_1m | FULL | 0.47 → 0.65 | 1.75 → 2.06 |
| B_bhar_1m | IS | 0.22 → 0.66 | 1.33 → 1.89 |
| B_bhar_1m | OOS | 0.41 → 0.79 | 0.36 → 1.39 |
| B_bhar_2m | FULL | **0.62 → 1.10** | **1.76 → 3.04** |
| B_bhar_2m | IS | **0.21 → 1.41** | **0.76 → 2.52** |
| B_bhar_2m | OOS | 0.82 → 0.63 | 1.69 → 1.26 |
| B_bhar_3m | FULL | 0.48 → 0.30 | 1.48 → 0.93 |
| B_bhar_3m | IS | 0.30 → 0.27 | 1.55 → 0.90 |
| B_bhar_3m | OOS | 0.28 → 0.17 | 0.37 → −0.43 |
| B_car_2m | FULL | 0.62 → 1.10 | 1.76 → 3.04 |
| B_car_2m | IS | 0.21 → 1.41 | 0.76 → 2.52 |
| B_car_2m | OOS | 0.82 → 0.63 | 1.69 → 1.26 |

## Assessment

The proper matched control **weakens** the headline B_bhar_2m result on FULL
and IS windows but **strengthens** several Signal-A OOS results
(A_bhar_{1,2,3}m OOS Sharpe rises sharply, alpha_t too). Two takeaways:

1. **B_bhar_2m FULL alpha_t falls from 3.04 → 1.76** and **IS alpha_t falls
   from 2.52 → 0.76**. The Day 4 sector-mean control was demonstrably too
   weak: it subtracted only the cross-sectional sector dispersion of recipients
   themselves, not the time-series drift of comparable non-recipients.
   Roughly 60% of the locked-in IS Signal B alpha was a sector-residualization
   artifact and is lost under matched control.

2. **A_bhar OOS unexpectedly strengthens.** The matched non-letter long leg
   underperformed the recipient short leg substantially in 2022-2024 — i.e.
   recipients did better than size+sector matches in the OOS window, the
   opposite of the IS direction. With Sharpe widening from 0.58 to 1.86 OOS
   for A_bhar_1m, this hints either at (a) a regime change post-2022, or
   (b) a methodological artifact from the price-only size proxy (matched
   firms could have lower market caps than the recipient and thus carry
   different small-cap risk loadings).

3. The `_car` and `_bhar` 2m signals coincide because per-month abnormal
   sums are identical when monthly returns are aggregated; this is consistent
   with Day 4.

## Files

- `scripts/day6_signal_matched.py` — builds matched factor returns
- `scripts/day6_compare.py` — head-to-head comparison printer
- `data/day6_factor_returns_matched.parquet` — output factor returns
- `data/day6_alpha_summary_matched.json` — orthogonalized alpha summary
- `data/day6_match_diagnostics.json` — per-signal match drop reasons + n_controls

## Limitations

- **Size proxy = adjusted close, not market cap.** Without shares-outstanding
  history this is the best PIT proxy in the existing data lake. ±20% of
  share price ≠ ±20% of market cap; firms with very different cap levels can
  share similar price bands. To upgrade: pull `sharesOutstanding` per ticker
  per filing date from yfinance's quarterly filings or Compustat FUNDQ.
- **Per-event K may dip below 5** in low-coverage sector-month cells (small
  sectors at the edges of the sample). Mean across all kept events is 4.88.
- **Letter exclusion uses upload_date only.** A firm with a corresp_date but
  no upload_date in the look-back window would still be in the control pool;
  in practice the dataset pairs UPLOADs and CORRESPs so this is a non-issue.
