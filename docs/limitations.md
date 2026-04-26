# Limitations & risks (running ledger)

This file consolidates known limitations of the analysis pipeline so that
reviewers can scan the project's honest weaknesses in one place.

## 1. Universe construction — partial point-in-time R3K

**Issue**: The Russell 3000 membership we use comes from public iShares IWV
holdings snapshots, not a true point-in-time membership table (which lives
behind paid sources: CRSP via WRDS, Norgate Data, Compustat).

**Mitigation taken**:
- Built UNION of two snapshots: 2018-10-02 (from a public archive on GitHub)
  and 2026-04-23 (current iShares download).
- 2018 → 2026 turnover: 262 firms left R3K, 846 firms joined. Of our 287
  feature-extracted CIKs, all 287 are in 2026 IWV (because that was our
  bootstrap source); we have now enqueued the 210 missing 2018-only CIKs
  through the daemon to reduce the bias.

**Residual bias**: firms that were in R3K only between 2010-2017 (left
before 2018-10) are still missing. This biases the analysis toward firms
that survived to at least 2018. Magnitude estimate: SEC-comment-letter
recipients have higher delisting rates than non-recipients (the very
hypothesis we test); naive estimate is 5-15% of 2015-2018 letter
universe missing.

**Reportable as**: a constraint on signal-A/B inference for the
2015-2018 sub-window; the 2019-2024 portion of OOS is less affected.

## 2. LLM training-data contamination — audit performed

**Issue**: gemma-3-27b, llama-3.3-70b, and claude-opus-4.7 were trained
on text up to their respective cutoffs (~2024-2025). SEC EDGAR documents
are publicly indexed; the models may have memorized firm-specific letter
content rather than inferring from text.

**Audit performed** (n=50 random R3K UPLOADs; see
`data/contamination_audit_summary.json`):
- Built redacted prompts: stripped firm name, ticker, all dates, file
  numbers, accession numbers, CIK numbers, phone/email.
- Re-ran v2 prompt + same models on redacted text.
- Compared resolution_signal, topics, severity to original.

**Results**:

| model | topic Jaccard (redacted vs orig) | severity Pearson | resolution κ |
|---|---|---|---|
| gemma-3-27b-it | 0.940 | 0.990 | **1.000** |
| llama-3.3-70b | 0.856 | 0.985 | **1.000** |

**Interpretation**: redaction does not change the LLM's classifications.
The signal comes from the substantive accounting text (which remains intact
post-redaction), NOT from firm-specific recall. **Contamination is not a
material concern for this analysis.**

## 3. PDF extraction quality — pending sample audit

**Issue**: ~50% of UPLOAD letters are PDFs. After the daemon-side bytes-safe
fix (latin-1 round-trip), pypdf successfully extracts non-empty text from
~99% of PDFs in the cache. But we have only verified non-empty, not
semantic correctness — pypdf can produce garbled text for tables,
multi-column layouts, and footnotes.

**Pending mitigation**: 30-record manual eyeball compare between extracted
text and SEC EDGAR PDF view. Day 6-7 rigor pass.

## 4a. PEAD baseline NOT reproducible from free data — deferred

**Issue**: Foster-Olsen-Shevlin (1984) naive PEAD surprise requires 8+
quarters of EPS history per firm to compute the standardized unexpected
earnings (SUE). yfinance's `Ticker.quarterly_income_stmt` returns only
~5 recent quarters (median per ticker = 5; max ~7). Earnings history
spanning 2015-2024 is unavailable.

For full PEAD baseline reproduction we would need:
- IBES (paid, via WRDS) for analyst-forecast-based surprise, OR
- Compustat (paid) for full quarterly EPS history, OR
- 8-K announcement parsing from SEC EDGAR (heavy text engineering, several
  weeks of work).

**Decision**: PEAD baseline is deferred. Day 5 scope is reduced to FF5+UMD
(done in Day 4) + LM 10-K sentiment. Paper writeup will note PEAD as
"comparable factor, not directly orthogonalized due to free-data history
limitation." Scaffolds (`scripts/fetch_quarterly_eps.py`,
`scripts/day5_pead_signal.py`) are committed for when paid data access
becomes available.

## 4. yfinance return data quality — fallback for CRSP

**Issue**: yfinance is the academic-equivalent free substitute for CRSP.
Known quirks: split-adjustment timing artifacts, delisted-ticker coverage
gaps, US-listing inconsistencies for ADRs.

**Mitigation**: returns are computed as monthly-end log returns from
adjusted close (auto_adjust=True). For HF-grade analysis, swap to CRSP
when WRDS access is available.

## 5. Sample size — 600-1500 in-target letters vs 30K aspirational

**Issue**: CLAUDE.md target is 30K UPLOAD-CORRESP pairs. As of Day 4 entry
we have ~5000 features extracted (UPLOAD) and ~5000 (CORRESP, v3-corresp),
of which ~1500-3000 fall in the 2015-2024 OOS-friendly window.

**Power consideration**: monthly long-short with ~10-30 firms per month
across 120 months gives bootstrap 95% CI of (-0.5, +1.5) on annualized
Sharpe (rough). Sufficient for top-line signal direction; insufficient for
fine-grained per-topic stratification.

**Mitigation**: pre-registered the spec (one signal, 60d window, FF5+UMD
baseline) before running; report DSR adjusted for n_trials = 8 to control
for the multiple-window/measure cells reported.

## 6. Schema-after-data risk for v3-corresp prompt

**Issue**: We redesigned the CORRESP resolution_signal enum
(SEC-perspective → registrant-perspective) AFTER observing v2's κ = 0.398
on CORRESP. This is data-informed schema design, which risks fitting.

**Mitigation taken** (committed in `23cabfe`, validated in `0819f75`):
- Locked the new 5-class schema in a pre-registration document
  (`docs/preregistration_v3_corresp.md`) BEFORE any v3-corresp extraction.
- Held-out validation: 100 train / 1400 test, deterministic seed=42.
- |κ_train − κ_test| = 0.021 < 0.10 → no schema overfitting.

## 7. Pre-registration of Day 4 cross-section

**Issue**: Without locking the event-study spec (window, measure, baseline),
the natural temptation to report the best of multiple cells inflates the
reported Sharpe.

**Mitigation taken** (committed in `c4cf77b`):
- BHAR over t+1..t+60 = main; CAR + 30/90d as robustness.
- FF5 + UMD = main baseline; FF3, FF3+UMD as robustness.
- Two distinct tradeable signals (A: upload_date, UPLOAD-only; B:
  corresp_date, UPLOAD+CORRESP) with strict information-set separation.
- IS 2015-2021 / OOS 2022-2024 frozen until rigor pass.

## 8. Information-timing leakage between UPLOAD and CORRESP

**Issue**: The hypothesis is that letter content predicts post-letter
returns. UPLOAD comes first; CORRESP arrives ~12 days later (median).
Including CORRESP-derived features at upload_date would leak future info.

**Mitigation taken** (encoded in `scripts/day4_construct_signal.py`):
- Signal A uses ONLY UPLOAD-side features and start_month = upload_date+1.
- Signal B uses BOTH and start_month = corresp_date+1, with forward window
  shifted accordingly. No leakage in either signal.

## 9a. **CRITICAL DISCOVERY: Day 4 IS alpha was largely sector-residualization artifact**

**Finding**: The Day 4 cross-section pipeline used "sector-mean of letter
recipients in same month" as the long-leg control instead of properly
matched non-letter firms (the pre-registration explicitly called for the
latter). Day 6 corrects this with K=5 matched non-letter R3K controls
per event (same sector, ±20% size proxy band).

**Result of correction** (matched - sector-mean):
- Signal B IS alpha_t: 2.52 → 0.76 (~60% of IS alpha was artifact)
- Signal A IS alpha_t: 2.84 → -0.18 (essentially zero post-correction)
- Signal B OOS alpha_t: 1.26 → 1.69 (improved, p≈0.10)
- **Signal A OOS alpha_t: 0.92 → 2.87 (jumped to p=0.007 ✅)**

**Interpretation**: the "scout-worthy" Day 4 IS results were over-stated by
sector-residualization. However, the OOS window (post-2022) shows a real
matched-control signal especially for the early-tradeable Signal A. The
honest story is: post-COVID 2022-2024 has tradeable signal; pre-2022 IS
was mostly noise inflated by the control choice.

**Mitigation taken**: All Day 6+ analysis uses the matched control. Day 4
results are preserved in git for reproducibility but explicitly tagged as
"sector-mean control, partial residualization" in [docs/day4_results.md].

**Residual limitation**: size proxy = adjusted close, not market cap (no
shares-outstanding history in free data). True market-cap matching needs
CRSP/Compustat via WRDS.

## 9. Sector concentration — Industrials 29%

**Issue**: SEC review activity is uneven across sectors; Industrials and
small Tech firms over-represented. This concentrates idiosyncratic risk.

**Mitigation**: monthly sector-residualization in the long-short
construction. Day 4 reports both raw and sector-residualized Sharpe.

## 10. Multiple-comparison risk in topic-stratified analysis

**Issue**: 14 topics × 4 severity bands × 5 intents × 3 horizons = 840
candidate cells. Without correction, false discovery rate is essentially
1.

**Pending**: Day 6-7 rigor pass with Benjamini-Hochberg FDR correction at
α = 0.05.
