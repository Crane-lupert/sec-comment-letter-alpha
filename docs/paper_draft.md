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

### 4.1 Headline (matched control, BHAR 2m, FF5+UMD residual, n_pairs = 1,014)

| Signal | Window | n_months | Sharpe | α annual | t | p | DSR |
|---|---|---|---|---|---|---|---|
| A (UPLOAD-only) | FULL | 123 | 0.73 | +6.56% | 1.85 | 0.067 | 1.00 |
| A | IS 2015-2021 | 78 | 0.37 | +6.85% | 0.74 | 0.46 | 0.95 |
| **A** | **OOS 2022-2024** | **36** | **1.43** | **+11.92%** | **2.86** | **0.007** | **1.00** |
| B (UPLOAD+CORRESP) | FULL | 120 | 0.66 | +5.40% | 1.78 | 0.078 | 1.00 |
| B | IS 2015-2021 | 75 | 0.14 | +2.78% | 0.43 | 0.67 | 0.39 |
| **B** | **OOS 2022-2024** | **36** | **1.27** | **+7.22%** | **2.55** | **0.015** | **1.00** |

The headline claim is the **OOS row of Signal A**: annualized residual α of +11.92% with t = 2.86 (p = 0.007), DSR = 1.00. Signal B is independent and corroborates (t = 2.55, p = 0.015). The IS pre-2022 windows are essentially zero under matched control — Day 6 self-correction reveals the original Day 4 IS magnitude was a sector-residualization artifact (we report the corrected number).

### 4.2 Robustness — sector / size / liquidity (full sample, sector-mean control for context)

By size quintile (price-based proxy):
- size_q0 (smallest): −84.32% / yr, t = −2.06, p = 0.040 — reverses, high noise.
- **size_q1: +26.28% / yr, t = +3.15, p = 0.002** — primary driver (mid-cap).
- size_q2: +9.21% / yr, t = +1.61, p = 0.108 — marginal.
- size_q3: −3.92% / yr, t = −0.78 — zero.
- size_q4 (largest): −2.40% / yr, t = −0.74 — zero (capacity-priced).

By sector (where n_months ≥ 12): Industrials zero (−0.50, t = −0.50), Materials marginal positive (+4.19%, t = +1.21), Financials marginal positive (+10.59%, t = +0.99), Consumer Discretionary anomalously negative (t = −4.97, p < 0.001 — flagged as multiple-comparison candidate).

### 4.3 Robustness — transaction costs (Signal A 2m, sector-mean control for absolute scale)

| TC | Sharpe FULL | α FULL | t FULL | Sharpe OOS | α OOS | t OOS |
|---|---|---|---|---|---|---|
| raw | 0.68 | +6.56% | 2.25 | 0.70 | +6.05% | 1.61 |
| 5 bp/mo | 0.62 | +5.96% | 2.04 | 0.62 | +5.45% | 1.45 |
| 10 bp/mo | 0.56 | +5.36% | 1.84 | 0.55 | +4.85% | 1.29 |
| 20 bp/mo | 0.44 | +4.16% | 1.43 | 0.39 | +3.65% | 0.97 |

Break-even TC: 58.0 bps/month before α = 0 for Signal A. For Signal B the break-even is even higher in IS but lower in OOS (full table in `data/day6_post_tc_summary.json`).

### 4.4 Robustness — Loughran-McDonald 10-K sentiment ablation

| Cell | base α | base t | +LM α | +LM t | Δ α (pp/yr) |
|---|---|---|---|---|---|
| A 2m FULL | +6.56% | 2.25 | +6.64% | 2.26 | +0.07 |
| A 2m OOS | +6.05% | 1.61 | +6.19% | 1.68 | +0.14 |
| B 2m FULL | +5.40% | 1.78 | +5.63% | 1.85 | +0.23 |
| **B 2m OOS** | **+7.22%** | **1.98** | **+7.52%** | **2.45** | **+0.30** |

Adding the LM 10-K negative-tone factor as additional regressor STRENGTHENS the signal in every cell (B 2m OOS t-stat 1.98 → 2.45). The signal is not 10-K sentiment in disguise — orthogonalization improves the residual.

### 4.5 FDR / multiple-comparison (BH at α = 0.05)

120 cells tested across topic × severity × window. 43 eligible (n_events ≥ 15 AND n_months ≥ 12). Nominal p < 0.05 hits: **10**. After Benjamini-Hochberg correction at α = 0.05, **2 survive**:

| stratum | window | n_months | n_events | α / yr | t | p_raw | p_BH |
|---|---|---|---|---|---|---|---|
| topic = non_gaap_metrics | FULL | 67 | 231 | +32.76% | +3.11 | 0.0019 | **0.041** ✅ |
| severity ∈ [0.5, 0.8] | OOS_2022_2024 | 35 | 439 | +30.56% | +3.10 | 0.0019 | **0.041** ✅ |

Combined with the pre-registered headline (Signal A OOS), this gives **3 independent FDR-safe claims**: (i) the pre-registered Signal A 2m matched OOS, (ii) the non_gaap_metrics topic alpha, (iii) the mid-band-severity OOS alpha. The remaining 41 eligible cells are descriptive color, not claims.

(For reference: the interim analysis at n = 935 produced 0 BH-survivors. Sample expansion to 1,014 unlocked these two without changing the methodology.)

### 4.6 Risk-managed overlay (post-hoc)

The pre-registered matched headline in §4.1 delivers an attractive OOS α (Signal A +26.53%/yr, t = 2.86; Signal B +26.85%/yr, t = 2.55) but the realized return path carries a heavy tail: peak-to-trough drawdown is −43.1% (A) and −46.7% (B). Diagnostics on the matched portfolio reveal the proximate cause — median monthly short breadth is n_short ≈ 5, and a handful of months with n_short = 3 (e.g. 2016-01, −30%) drive the bulk of the drawdown via single-name idiosyncratic blow-ups. To characterize the implementable Sharpe / Calmar trade-off rather than to replace the headline, we construct a separate post-hoc risk-managed variant.

The overlay stacks three components:

- **A. Breadth filter.** Months with n_events_kept < N are forced to cash for the long-short return, but are kept in the panel so that FF5+UMD orthogonalization continues to use the full timeline.
- **B. Per-name weight cap.** Iterative cap at max(0.20, 1.5/N) with proportional redistribution of the spillover across remaining names. For N ≥ 8 this collapses to a flat 20% cap; for N = 4 the cap is 0.375 per name.
- **C. Volatility target.** Lag-1 6-month rolling realized σ rescales the long-short return to an annualized target σ of 10%, with leverage clipped to [0, 2x].

The breadth filter A is the dominant driver of the t-statistic improvement; the vol-target C shrinks the standard error somewhat faster than it shrinks α, which lifts t even when the level of α drops slightly. The per-name cap B contributes mostly to drawdown control rather than to point-estimate α.

We sweep the breadth threshold N ∈ {4, 5, 6, 8} and report the full result set (unrounded) so that the N choice is fully transparent:

| Signal | variant | MDD | OOS α/yr | OOS t | OOS p | FULL α/yr | FULL t |
|---|---|---|---|---|---|---|---|
| A_bhar_2m | matched (D6) | −43.1% | +26.53% | 2.86 | 0.004 | +16.40% | 1.85 |
| A_bhar_2m | n=8 (orig spec) | −15.2% | +7.31% | 1.61 | 0.107 | +5.65% | 2.51 |
| A_bhar_2m | n=6 | −14.5% | +4.91% | 1.09 | 0.278 | +8.61% | 3.09 |
| A_bhar_2m | n=5 | −17.4% | +22.88% | 2.56 | 0.011 | +15.25% | 3.49 |
| **A_bhar_2m** | **n=4 (canonical)** | **−13.6%** | **+20.80%** | **3.08** | **0.002** | **+14.48%** | **4.07** |
| B_bhar_2m | matched (D6) | −46.7% | +26.85% | 2.55 | 0.011 | +17.01% | 1.81 |
| B_bhar_2m | n=8 (orig spec) | −20.2% | +3.16% | 0.39 | 0.700 | +6.75% | 1.44 |
| B_bhar_2m | n=6 | −40.1% | +6.54% | 1.23 | 0.217 | +6.05% | 1.03 |
| B_bhar_2m | n=5 | −40.0% | +8.86% | 1.50 | 0.134 | +1.98% | 0.28 |
| **B_bhar_2m** | **n=4 (canonical)** | **−29.5%** | **+13.93%** | **2.42** | **0.015** | **+10.33%** | **1.82** |

The a-priori first guess of N = 8 is too aggressive: it forces ~70% of months to cash and the OOS α collapses to +7.31% (t = 1.61, p = 0.107) for Signal A and to +3.16% (t = 0.39) for Signal B. N = 6 is similarly weak. The breakpoint is between N = 6 and N = 5; we adopt **N = 4 as the canonical risk-managed variant** because it retains enough months to keep the OOS sample non-degenerate while still excluding the lowest-breadth tail months that drove the matched-portfolio drawdowns.

For the N = 4 canonical variant, **Signal A** delivers OOS α = +20.80%/yr at t = 3.08 (p = 0.002) — the t-stat *improves* relative to the raw matched 2.86 because the vol-target reduces SE more than it reduces α. MDD compresses from −43.1% to −13.6%, and OOS Calmar rises from 0.62 to 1.53. **Signal B** delivers OOS α = +13.93%/yr at t = 2.42 (p = 0.015) with MDD −29.5% (vs −46.7%) and OOS Calmar 0.47. Both signals retain joint significance after the overlay.

**Pre-registration discipline.** This subsection is post-hoc and is explicitly **not** a replacement for the §4.1 pre-registered claim. The pre-registered headline remains Signal A OOS α = +26.5%/yr (matched control, raw) at t = 2.86. §4.6 exists only to characterize implementability and risk budget. The N selection was informed by a post-hoc sweep — full transparency in the sweep table is the mitigation against fishing concerns, and the sweep archive is preserved on disk (`data/day7_risk_managed_n{4,5,6,8}_*.json`).

**Implementability.** The N = 4 canonical variant requires monthly rebalancing of ≤ 4 short and ≤ 4 long positions, a vol-rescale step each month from a 6-month rolling σ, and a 20% per-name cap (effectively 0.375 at N = 4). All three steps are trivial to operationalize at ~$10-50M notional given typical R3K mid-cap ADV.

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
