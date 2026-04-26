# SEC Comment Letter Cross-Section Alpha — paper draft outline

**Status**: Day 11 outline. Numbers will be filled in after Day 6 sample expansion completes and the matched-control re-run finishes.

---

## Title (working)

**"SEC Comment Letter Disclosure Vulnerability and Post-Letter Returns: A Pre-Registered Cross-Section Test"**

---

## Abstract (~150 words, draft)

Using a multi-LLM ensemble (gemma-3-27b, llama-3.3-70b, claude-opus-4.7) to extract topic, severity, and registrant-response intent from N=XXX SEC UPLOAD-CORRESP letter pairs across the Russell 3000 from 2015 to 2024, we test whether the comment-letter event predicts post-letter abnormal returns. After pre-registration of the event-study spec (commit `c4cf77b`) — BHAR over t+1..t+60 days, FF5+UMD residual, IS 2015-2021 / OOS 2022-2024 frozen — and a held-out validation of the registrant-perspective schema (κ_train 0.876 vs κ_test 0.856), we find that an early-tradeable signal (UPLOAD-only at upload_date) yields an annualized residual alpha of XX% (t=XX) in OOS under matched-non-letter control, concentrated in the size_q1-q2 mid-cap quintiles. The signal is independent of FF5+UMD+LM-10K-sentiment, contaminates audit (κ=1.000) does not invalidate, and survives 20bp/month transaction cost (Sharpe XX, t=XX). Per-topic and per-severity stratifications do not survive Benjamini-Hochberg FDR at α=0.05; the headline DSR-corrected signal is the only valid claim.

---

## 1. Introduction (~1 page)

- Hypothesis: SEC comment letters reveal firm-level disclosure vulnerabilities. Letter content predicts post-letter abnormal returns via a (topic × severity × response-pattern) cross-sectional structure.
- Prior art: We are not aware of any LLM-based extraction on the UPLOAD-CORRESP **pair** (i.e. dialog) for cross-section signal construction. Prior work focuses on text-mining 10-Ks (Loughran-McDonald), or single-letter narrative scoring.
- Contribution: (1) a pre-registered ensemble pipeline with held-out schema validation; (2) explicit information-set separation (Signal A early-tradeable at upload_date vs Signal B late-tradeable at corresp_date); (3) concrete out-of-sample post-cost Sharpe under matched-non-letter control; (4) honest negative findings (sector-residualization artifact, multiple-comparison hunting via FDR).

## 2. Data (~1.5 pages)

### 2.1 Universe
- 2,790 R3K members from PIT-union of iShares IWV (2018-10 + 2026-04) holdings.
- Ticker → CIK via SEC company_tickers.json one-shot static metadata.
- 19,639 R3K UPLOAD filings + 18,386 R3K CORRESP filings cached via daemon.

### 2.2 Letter pairing
- UPLOAD ↔ CORRESP within 90-day window per CIK (median lag 12 days, p90 28 days).
- 12,883 matched pairs total.

### 2.3 Returns
- Monthly adjusted-close log returns from yfinance (R3K 2522 of 2528 tickers).
- Forward-return horizons: t+1..t+30, t+1..t+60 (main), t+1..t+90.
- BHAR (main) and CAR (robustness).

### 2.4 Baselines
- Fama-French 5-factor + Carhart UMD (Kenneth French Data Library), monthly.
- Loughran-McDonald 10-K negative-tone factor (5,285 R3K 10-Ks, 491 firms).

### 2.5 Limitations (forward reference)
- Survivorship bias from PIT-union (pre-2018 R3K-only firms missing).
- Size proxy = adjusted close (no shares-outstanding history without WRDS).
- yfinance vs CRSP standard.
- See `docs/limitations.md` for the full ledger.

## 3. Method (~2 pages)

### 3.1 Feature extraction
- Ensemble: gemma-3-27b + llama-3.3-70b (both via OpenRouter), with claude-opus-4.7 oracle on n=30.
- v2 prompt for UPLOAD: 14-class topic enum + 0-1 severity + 4-class resolution_signal.
- v3-corresp prompt for CORRESP: same topics + severity + 5-class registrant response_intent (agree_revise / explain_position / supplemental / pushback / closing).
- Schema design pre-registration: `docs/preregistration_v3_corresp.md`.
- Held-out validation: 100 train / 1400 test, deterministic seed=42. |κ_train − κ_test| = 0.021.

### 3.2 Inter-rater reliability
- gemma vs llama: topic Jaccard 0.79, severity Pearson 0.97, resolution κ 0.93 (UPLOAD); 0.74 / 0.89 / 0.86 (CORRESP v3-corresp full sample).
- Three-model oracle on n=30 (v2+opus): mean κ 0.72.

### 3.3 Contamination audit
- 50 random UPLOADs with redacted firm name / date / file numbers / accession.
- gemma κ(redacted vs original) = 1.000, llama κ = 1.000.
- Topic Jaccard 0.94 / 0.86, severity Pearson 0.99 / 0.99.
- Conclusion: LLM features come from substantive text content, not firm-specific recall.

### 3.4 Cross-section signal construction
- Pre-registered (commit `c4cf77b`):
  - Signal A: event=upload_date, UPLOAD-only features, forward window upload_date+1 to +60.
  - Signal B: event=corresp_date, UPLOAD+CORRESP features, forward window corresp_date+1 to +60.
- Severity-weighted short on letter recipients each calendar month.
- Long control: K=5 matched non-letter R3K firms (same sector, ±20% size proxy band; price-based proxy because no shares-outstanding history). Equal-weight long basket.
- Day 4 used "sector-mean of recipients" as control (over-stated by sector-residualization). Day 6 corrects with proper matched control.

### 3.5 Orthogonalization & inference
- Monthly long-short return regressed on FF5 + UMD; intercept = α.
- Newey-West HAC SE, lag = 6.
- Month-clustered bootstrap 95% CI (B = 1000).
- Bailey-López de Prado Deflated Sharpe Ratio with n_trials = 8 (pre-registered cells).
- Benjamini-Hochberg FDR at α = 0.05 for any per-stratum analysis.

### 3.6 Pre-registration discipline
- Three commit-locked pre-registrations:
  - `23cabfe` — v3-corresp schema before extraction.
  - `0819f75` — held-out validation outcome.
  - `c4cf77b` — Day 4 cross-section spec before run.

## 4. Results (~2 pages — TO FILL POST-EXPANSION)

### 4.1 Headline (matched control, BHAR 2m, FF5+UMD residual)
| Signal | Window | n_months | Sharpe | α annual | t | p | DSR |
|---|---|---|---|---|---|---|---|
| A (UPLOAD-only) | FULL | XX | XX | XX% | XX | XX | XX |
| A | IS 2015-2021 | XX | XX | XX% | XX | XX | XX |
| **A** | **OOS 2022-2024** | **XX** | **XX** | **XX%** | **XX** | **XX** | **XX** |
| B (UPLOAD+CORRESP) | FULL | XX | XX | XX% | XX | XX | XX |
| B | IS | XX | XX | XX% | XX | XX | XX |
| B | OOS | XX | XX | XX% | XX | XX | XX |

### 4.2 Robustness — sector / size / liquidity
- Signal concentrated in size_q1-q2 mid-cap quintiles.
- Smallest quintile reverses (likely high-noise low-volume).
- Largest quintile zero alpha (capacity-constrained or already-priced).
- Sector breakdown: Industrials zero, Materials/Financials marginal positive.

### 4.3 Robustness — transaction costs
- 5/10/20 bps/month: Signal B retains XX/XX/XX% alpha annual; Sharpe XX/XX/XX.
- Break-even TC: XX bps/month before alpha = 0.

### 4.4 Robustness — LM 10-K sentiment ablation
- B alpha shifts -0.25pp (FULL) when LM added as additional regressor; t-stat unchanged (3.04 → 3.05).
- OOS B alpha actually improves (1.26 → 1.53) with LM control.
- Conclusion: signal is independent of 10-K negative-tone factor.

### 4.5 FDR / multiple-comparison
- 120 cells tested across topic × severity × window.
- 40 eligible (n_events ≥ 15 AND n_months ≥ 12).
- Nominal p<0.05: 3 (chance-rate of 2 expected).
- BH FDR survivors: 0.
- Conclusion: pre-registered cells are the only valid claims; topic-level slicing is descriptive color.

## 5. Discussion (~1 page)

### 5.1 Why does Signal A outperform Signal B in OOS?
- Hypothesis: post-2022 the UPLOAD-only severity feature captures the disclosure vulnerability earlier; CORRESP arrives ~12 days later by which point the market has partially priced in.
- Alternative: Day 6 matched control happens to favor early signal in 2022-2024 due to sector composition shifts.

### 5.2 Why is the signal size-concentrated?
- Mid-cap firms have less analyst coverage → SEC comment letters carry more news content.
- Largest firms already extensively analyzed → letter is non-news.
- Smallest firms have noise dominating signal.

### 5.3 What happened to the IS alpha?
- Day 4 reported Signal B IS Sharpe 1.41, t 2.52. Day 6 matched control reveals this was largely sector-residualization artifact (Sharpe 0.21, t 0.76 post-correction).
- Honest reframing: the signal is a 2022+ phenomenon, not a 2015-2021 phenomenon. May indicate post-COVID market's response to disclosure vulnerability is sharper.

### 5.4 What's the capacity?
- Mid-cap concentration suggests capacity ~$50M-$200M in long-short notional given typical R3K mid-cap ADV.
- Larger fund deployment would erode alpha via market impact.

## 6. Limitations & threats to validity (~0.5 page)

- (See `docs/limitations.md` for the full ledger.)
- Survivorship from PIT-union universe.
- Size proxy quality (price-based, not market-cap).
- yfinance return data quality vs CRSP.
- 935 → XXX pairs is below CLAUDE.md 30K target; statistical power bounded.
- v3-corresp schema redesigned post-v2 (held-out validation mitigates but does not eliminate fitting risk).
- Per-topic FDR null result is honest; could mean topic dimension is uninformative OR sample too small.

## 7. Reproducibility (~0.5 page)

- Public GitHub repo + Streamlit dashboard.
- 1-command reproduction: `scripts/day4_run_all.py` after extraction.
- Pre-registration trail in git log: `git log --oneline | grep "Pre-register"`.
- All data files (gitignored) regenerable from the daemon-cached SEC filings.

## 8. References (TBD)

- Loughran & McDonald (2011)
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio
- Newey & West (1987) — HAC SE
- Foster, Olsen, Shevlin (1984) — PEAD naive surprise
- Fama & French (2015) — 5-factor model
- Carhart (1997) — Momentum (UMD)
- Benjamini & Hochberg (1995) — FDR

---

## Notes for final fill-in (post-Day 6 sample expansion)

- Replace XX placeholders with final numbers from data/day4_alpha_summary.json (matched variant) and data/day6_post_tc_summary.json.
- Re-run scripts/day7_robustness.py and scripts/day7_fdr.py with the matched control on the expanded sample (15K UPLOAD + 15K CORRESP_v3).
- Update Section 4.1 table.
- Verify Section 4.5 still null after expansion (if any cell suddenly survives BH, that's a new finding worth documenting).
- Update sample-size mention from 935 → final pair count.

## Word/page budget

- Target: 8-12 pages SSRN-grade (excluding tables and references).
- Current outline: ~7 pages dense or ~10 pages typeset.
- Iteration cycle: write Section 4 fill-in → review → trim discussion if over.
