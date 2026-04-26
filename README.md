# SEC Comment Letter Cross-Section Alpha

**Status**: 14-day timebox complete (2026-04-27). [GitHub](https://github.com/Crane-lupert/sec-comment-letter-alpha) + [Streamlit dashboard](https://sec-comment-letter-alpha-260427ah.streamlit.app) live. **한국어 실무 보고**: [README_KR.md](README_KR.md).

## What this is

Cross-sectional equity alpha from ~5,000 R3K SEC `UPLOAD` (SEC → company) and `CORRESP` (company → SEC) letter pairs, 2015-2025, via a multi-LLM ensemble (gemma-3-27b + llama-3.3-70b + claude-opus-4.7 oracle). Pre-registered, held-out validated, contamination-audited. Final analyzable sample: 1,014 letter pairs.

## 5-minute skim

1. **Hypothesis**: SEC comment letters reveal firm-level disclosure vulnerabilities; letter content predicts post-letter abnormal returns through (topic × severity × response intent).
2. **Method**: ensemble extraction → severity-weighted long-short with K=5 matched non-letter R3K controls (same sector, ±20% size band) → FF5+UMD residual α + Newey-West HAC + month-cluster bootstrap CI + Bailey-López de Prado DSR.
3. **Headline (matched control, BHAR 2m, n=1,014 pairs)**:
   - **Signal A** (UPLOAD-only, early-tradeable) **OOS 2022-2024**: α=**+11.92%/yr**, **t=2.86**, p=0.007, **DSR=1.00**, Sharpe 1.43, n_OOS=36 months.
   - **Signal B** (UPLOAD+CORRESP, late-tradeable) OOS: α=+7.22%/yr, t=2.55, p=0.015, DSR=1.00, Sharpe 1.27.
   - Honest finding: ~60% of Day 4 IS alpha was sector-residualization artifact (self-corrected on Day 6, both versions preserved). See [docs/day14_final_results.md](docs/day14_final_results.md) and [docs/day6_matched_control.md](docs/day6_matched_control.md).
4. **Robustness**: TC break-even ~71 bps/month (3-15× margin over realistic 5-20 bps). LM 10-K sentiment ablation: signal independent (Signal B OOS t 1.98 → 2.45 with LM in baseline). Mid-cap (size_q1) concentration: +26%/yr t=3.15; large-cap zero alpha. **BH FDR survivors** (43 cells, α=0.05): 2 — non_gaap_metrics FULL (α=+33%, p_BH=0.041) + severity 0.5-0.8 OOS (α=+31%, p_BH=0.041). Plus pre-registered headline = **3 FDR-safe claims**.
5. **Reproducibility**: pre-registration commits `c4cf77b`, `23cabfe`, `0819f75`. 1-command run: `scripts/day4_run_all.py`.

## Repo structure

```
src/sec_comment_letter_alpha/
  data_loader.py    # daemon-mediated SEC cache reader
  parse.py          # text → structured segments + UPLOAD-CORRESP pairing
  features.py       # ensemble LLM extraction (v1, v2, v3-corresp prompts)
  llm.py            # OpenRouter wrapper (single-attempt + own backoff)
  pipeline.py       # CLI: enqueue / status / dry-run / ensemble
  stats.py          # FDR, DSR, bootstrap CI helpers
  universe.py       # R3K parquet loader (PIT-union default)

scripts/
  bootstrap_universe_r3k.py     # IWV + SEC ticker map → R3K universe
  fetch_yfinance_returns.py     # monthly returns for R3K tickers
  fetch_french_factors.py       # FF5 + UMD from Kenneth French
  fetch_lm_dictionary.py        # Loughran-McDonald master dictionary
  fetch_quarterly_eps.py        # quarterly EPS (Day 5 PEAD scaffold)
  day3_extract.py               # UPLOAD ensemble extraction (resumable, parallel)
  day3_corresp_extract.py       # CORRESP ensemble extraction
  day3_corresp_v3_extract       # via day3_corresp with --prompt-version v3-corresp
  day4_build_pairs.py           # UPLOAD-CORRESP join → 12-dim feature pairs
  day4_build_panel.py           # firm-month panel + per-event BHAR/CAR
  day4_construct_signal.py      # severity-weighted long-short factor (sector-mean control)
  day4_orthogonalize.py         # FF5+UMD residual α + NW + bootstrap + DSR
  day4_run_all.py               # orchestrator
  day5_lm_sentiment.py          # LM 10-K negative-tone factor
  day5_ablation_lm.py           # joint regression Signal vs FF5+UMD+LM
  day5_pead_signal.py           # PEAD scaffold (deferred - free EPS too short)
  day6_apply_tc.py              # post-cost Sharpe at 5/10/20 bps/month
  day6_signal_matched.py        # K=5 matched non-letter long control
  day6_compare.py               # matched vs sector-mean side-by-side
  day6_pdf_audit.py             # PDF extraction quality heuristics
  day7_robustness.py            # sector / size / liquidity stratified α
  day7_fdr.py                   # BH FDR on per-topic stratification
  contamination_audit.py        # redacted-vs-original κ check
  bootstrap_universe.py         # legacy SEC-tickers universe (Day 1)
  build_corresp_v3_split.py     # held-out train/test split for v3-corresp

docs/
  preregistration_v3_corresp.md   # CORRESP schema lock
  preregistration_day4_event_study.md   # event-study spec lock
  preregistration/v3_corresp_results.md # held-out validation outcome
  day4_results.md                # Day 4 headline (sector-mean control)
  day6_matched_control.md        # Day 6 matched headline (corrects Day 4)
  day6_pdf_audit.md              # PDF extraction trustability
  day7_robustness.md             # size / sector / liquidity strata
  day7_fdr.md                    # BH FDR null result
  paper_draft.md                 # Day 11 paper outline (XX placeholders)
  limitations.md                 # running ledger of 11 known limitations
  overnight_2026-04-26_summary.md  # mid-overnight scout-readiness checkpoint

dashboard/
  app.py                         # Streamlit; 6 tabs (overview, heatmap, etc.)
  README.md                      # run + deploy instructions

tests/
  test_smoke.py                  # parse + agreement metric tests (17)
  test_day6_matched.py           # matching logic tests (7)
  test_day7_fdr.py               # BH FDR tests (4)
```

28 tests passing.

## Setup

```bash
cd D:/vscode/sec-comment-letter-alpha
uv venv
uv pip install -e .
uv pip install -e D:/vscode/portfolio-coordination/shared-utils
copy .env.example .env  # confirm PORTFOLIO_COORD_ROOT
```

The SEC daemon (in `D:/vscode/portfolio-coordination/`) populates the cache; this repo never calls SEC directly.

## Run end-to-end

```bash
# 1. Universe + market data
.venv/Scripts/python.exe scripts/bootstrap_universe_r3k.py
.venv/Scripts/python.exe scripts/fetch_yfinance_returns.py
.venv/Scripts/python.exe scripts/fetch_french_factors.py
.venv/Scripts/python.exe scripts/fetch_lm_dictionary.py

# 2. Enqueue daemon for R3K UPLOAD/CORRESP fetch
.venv/Scripts/python.exe -m sec_comment_letter_alpha.pipeline enqueue --n 1000 --seed 17
# (wait for daemon to fetch -- monitor with .venv/Scripts/python.exe -m sec_comment_letter_alpha.pipeline status)

# 3. LLM ensemble extraction (~$3-5 OpenRouter)
.venv/Scripts/python.exe scripts/day3_extract.py --n 5000 \
    --output data/day3_features_r3k.jsonl \
    --universe-filter data/universe_ciks_r3k.parquet \
    --record-parallelism 4

.venv/Scripts/python.exe scripts/day3_corresp_extract.py --n 5000 \
    --output data/day3_corresp_v3_full.jsonl \
    --universe-filter data/universe_ciks_r3k.parquet \
    --prompt-version v3-corresp \
    --record-parallelism 4

# 4. Cross-section pipeline (~5 min)
.venv/Scripts/python.exe scripts/day4_run_all.py

# 5. Day 6 corrections (matched control + TC)
.venv/Scripts/python.exe scripts/day6_signal_matched.py
.venv/Scripts/python.exe scripts/day4_orthogonalize.py \
    --input data/day6_factor_returns_matched.parquet \
    --output data/day6_alpha_summary_matched.json
.venv/Scripts/python.exe scripts/day6_apply_tc.py

# 6. Day 7 robustness + FDR
.venv/Scripts/python.exe scripts/day7_robustness.py
.venv/Scripts/python.exe scripts/day7_fdr.py

# 7. Day 5 LM ablation
.venv/Scripts/python.exe scripts/day5_lm_sentiment.py
.venv/Scripts/python.exe scripts/day5_ablation_lm.py

# 8. Dashboard
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

## Pre-registration discipline

Three commit-locked pre-registrations:

| Commit | Document | Purpose |
|---|---|---|
| `23cabfe` | docs/preregistration_v3_corresp.md | CORRESP schema + held-out split, locked BEFORE v3-corresp extraction |
| `0819f75` | docs/preregistration/v3_corresp_results.md | Validation outcome (κ_train 0.876, κ_test 0.856, gap 0.021 < 0.10) |
| `c4cf77b` | docs/preregistration_day4_event_study.md | Day 4 event-study spec (BHAR 2m, FF5+UMD, IS/OOS frozen, 8-cell DSR) |

## Honest acknowledgments

- **Day 4 IS alpha was largely sector-residualization artifact.** Day 6 matched-control correction reveals the signal as primarily an OOS-2022-2024 phenomenon.
- **Per-topic stratification fails BH FDR at α=0.05** (3 nominal hits / 40 cells, 0 BH-survivors). The headline DSR-corrected alpha is the only valid claim.
- **Universe survivorship**: PIT-union of 2018 + 2026 IWV holdings; pre-2018 R3K-only firms missing.
- **Size proxy = adjusted close** (price-based), not market cap; for true cap matching needs WRDS access.
- **PEAD baseline deferred**: yfinance EPS history insufficient (5 quarters only).

See [docs/limitations.md](docs/limitations.md) for the full ledger.

## Cost / performance budgets

- OpenRouter (project X): ~$XX / cap $45 (final post-expansion).
- Wall time: ~8 hours overnight from raw daemon-cache to publishable result.
- All daemon fetches centralized through `D:/vscode/portfolio-coordination/scripts/sec-agent-daemon.py` (per-doc rate-limited at 8 RPS).

## Repo philosophy

Open-source, reproducible, pre-registered. If a finding can be cherry-picked, it isn't reported as a finding. If a measurement can be deflated, it is.

---

Project X of QR Scout Portfolio. See `CLAUDE.md` for the day-by-day execution plan + advance gates + abandon criteria.
