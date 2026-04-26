# 5-minute interview demo script

Use as a verbal walkthrough alongside the live Streamlit dashboard
(`streamlit run dashboard/app.py`).

---

## (0:00 — 0:30) Hook

> "I built a cross-sectional equity alpha from ~5,000 SEC comment letter pairs using a multi-LLM ensemble (gemma + llama + claude). The signal is pre-registered, held-out validated, contamination-audited, and survives FF5+UMD+LM-sentiment orthogonalization at 10 bps/month transaction cost."

If the interviewer is interested in mechanics, the next 4 minutes drill down. If the interviewer asks a specific question, jump to the relevant tab.

---

## (0:30 — 2:00) Method walkthrough — open Dashboard "Overview"

> "SEC publishes UPLOAD letters when they comment on a firm's filing. The firm replies via CORRESP. There's a public dialog. My hypothesis: the topic and severity of these letters reveal disclosure vulnerabilities the firm hasn't fully resolved, and that has cross-sectional return predictability."

Point at headline numbers:

> "After matched-non-letter controls and FF5+UMD residual, Signal A — the early-tradeable version, using only UPLOAD content at the upload date — produces an out-of-sample annualized alpha of about XX%, t-stat 2.87. Pre-registered, held-out validated."

Click the "Use Day 6 matched control" toggle off:

> "Notice the older Day 4 numbers were inflated. I caught a methodology issue in my own work: I had been using sector-mean of letter recipients as the long-leg control, which residualizes sector-driven covariance into the signal. When I switch to K=5 matched non-letter R3K firms in the same sector and ±20% size band, ~60% of the IS alpha disappears — and OOS actually strengthens. Self-correction visible in git log."

---

## (2:00 — 3:00) Robustness — click "Robustness" tab

> "Five robustness layers. First, transaction costs: the signal break-even is around 70 bps per month, so realistic 10-20 bps barely move the t-statistic."

Click "TC sensitivity" tab; show the 10bp row.

> "Second, baseline orthogonalization. I added Loughran-McDonald 10-K negative-tone factor to the FF5+UMD baseline; the signal's alpha barely shifts. So this isn't 10-K sentiment in disguise."

Click "Robustness" tab; point at size-quintile rows.

> "Third, size stratification. The signal concentrates in size_q1-q2 — mid-cap. Largest firms show zero alpha (consistent with capacity-constrained, already-priced). Smallest reverses with high noise. Reasonable capacity guess: $50M-$200M long-short notional."

---

## (3:00 — 4:00) Honest negative findings — click "About"

> "Three things I want to highlight that I caught and reported myself, not buried."

Show docs/day7_fdr.md:

> "First: per-topic and per-severity stratification — 40 eligible cells — fails Benjamini-Hochberg FDR at α=0.05. Three nominal p<0.05, zero survive correction. The headline DSR-corrected number is the only valid claim. Anyone slicing further is multiple-comparison hunting."

Show contamination audit summary:

> "Second: contamination audit. I redacted firm name, dates, and file numbers from 50 random letters and re-ran the LLM. κ between redacted and original was 1.000 — the model is reading text content, not retrieving memorized firm-specific outcomes. Important for any LLM-based research right now."

Show docs/limitations.md:

> "Third: limitations document — 11 known issues. Universe survivorship from PIT-union, size proxy quality, yfinance vs CRSP, sample size below the aspirational 30K target. Each one with a mitigation taken or a path to fix with paid data access."

---

## (4:00 — 4:30) Pre-registration

> "Three pre-registration commits, locked before extraction. Schema design before v3-corresp re-run. Held-out validation: 100 train / 1400 test, deterministic seed, |κ_train − κ_test| = 0.021 — the schema isn't fitted to data I'd already seen. Day 4 event-study spec — BHAR window, baseline, IS/OOS — locked before any cross-section run. Git log proves the timeline."

---

## (4:30 — 5:00) Repro + close

> "Open-source repo, 1-command reproduction after the daemon cache is populated. Streamlit dashboard publicly hosted. SSRN paper draft committed (Day 11). I'd characterize this as a preliminary scout-grade signal that needs WRDS-grade data and a longer OOS window to land in fund production. The methodology — pre-registration, ensemble + audit, matched controls, FDR-safe inference — is the actual contribution."

Pause. Wait for follow-up question.

---

## Anticipated follow-up questions

| Question | One-line answer |
|---|---|
| "Why this signal and not 10-K sentiment?" | "10-K sentiment is annual + retrospective; SEC letters are quarterly + future-vulnerability. Ablation shows independent." |
| "What's the capacity?" | "Mid-cap concentration suggests $50-200M long-short notional. Larger erodes alpha via impact." |
| "Why is OOS stronger than IS?" | "Honest answer: I don't know yet. Two hypotheses: post-COVID disclosure regime shift, or matched-control surfaces signal that sector-mean residualization had washed out. Need more OOS time." |
| "How long to deploy in production?" | "Daily/weekly extraction is feasible — daemon already enqueues new filings as they appear. Monthly rebalance for now; intra-month event-driven is a research extension." |
| "Why ensemble over a single big model?" | "κ between models is the IRR check on the LLM extraction itself. Single model = no fail-safe. Ensemble = fail-safe + DSR-correctable." |
| "What if the LLM saw the data?" | "Contamination audit answered: features are inferred from text content, not recalled. κ=1.000 between redacted and original." |
| "Show me a single trade." | Click Topic-heatmap → revenue_recognition + severity 0.5-0.8 → drill to event list → pick one. Walk through letter date, severity, sector, post-letter return. |
