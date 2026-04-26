# Day 6 → Day 13 self-audit (mid-overnight)

**Date**: 2026-04-26
**Sample state**: 7155 R3K UPLOAD features + 6057 R3K CORRESP v3-corresp features (extraction in flight to 15000 each, ETA ~4h).

## Day 6 — rigor pass (CLAUDE.md gate: "all rigor test passing OR explicit limitations")

| # | Check | Status | Reference |
|---|---|---|---|
| 6.1 | Transaction cost model | ✅ done — break-even 71.6 bps/mo Signal B; 5/10/20bp scenarios reported | `scripts/day6_apply_tc.py`, commit `d2542f1` |
| 6.2 | Sector-matched non-letter control | ✅ done — uncovered IS sector-residualization artifact; OOS Signal A t=2.87 | `scripts/day6_signal_matched.py`, commit `7914389` |
| 6.3 | Multiple-comparison FDR (BH) | ✅ done — 0/40 cells survive at α=0.05; per-topic null | `scripts/day7_fdr.py`, commit `7ac7d20` |
| 6.4 | PDF extraction quality audit | ✅ done — TRUSTABLE (full+partial = 100%, garbled 0%, mean conf 0.947) | `scripts/day6_pdf_audit.py`, commit `7ac7d20` |
| 6.5 | Contamination audit | ✅ done at Day 6 prep — κ=1.000 redacted vs original | `scripts/contamination_audit.py` |
| 6.6 | Sample expansion 1500 → 15000 | ⏳ in flight — UPLOAD 7155, CORRESP_v3 6057; ETA ~4h | extraction logs |

**Day 6 verdict**: ✅ **PASS with explicit limitations**.

## Day 7 — additional rigor (CLAUDE.md: sector/size/liquidity, residual α CI)

| # | Check | Status |
|---|---|---|
| 7.1 | Sector / size / liquidity stratified α | ✅ size_q1-q2 mid-cap concentration |
| 7.2 | Newey-West HAC SE (lag=6) | ✅ applied |
| 7.3 | Bootstrap CI (B=1000, monthly cluster) | ✅ in day4_alpha_summary.json |
| 7.4 | Contamination audit | ✅ done at Day 6 |
| 7.5 | Residual α CI 0-inclusion check | ✅ IS excludes 0; OOS includes 0 |

**Day 7 verdict**: ✅ **PASS**. Limitations: size proxy = price (not market cap); LM 10-K factor uses 491 of 2528 firms.

## Day 8 — dashboard

✅ Scaffold ready, 6 tabs coded, toggle Day 4 ↔ Day 6 matched. Deployment pending user action.

## Day 11 — paper draft

🟡 Outline locked, numbers pending final extraction.

## Day 12 — README

✅ Done. Commit `ad0f6f3`.

## Day 13 — interview demo

✅ Done. 5-minute script with dashboard click flow + 7 anticipated Q&A. Commit `ad0f6f3`.

## Day 14 — buffer

⏳ Pending. Tasks:
- Final-data re-run of all Day 4-7 scripts → fill XX placeholders in paper
- Final pytest run (target: still 28+ passing)
- Streamlit dashboard hosted screenshots (5 use cases)
- README update with final numbers
- Final overnight checkpoint

## Cost / budget

OpenRouter project X spend at this audit point: ~$5.85 / $45 cap (13%). Final post-expansion projection: ~$12-15 / $45 cap (33%).

## What scout-grade looks like vs where we are

**Scout-grade requirements**:
- ✅ Pre-registered (3 commits)
- ✅ Held-out validated (gap 0.021 < 0.10)
- ✅ Contamination-audited (κ=1.000)
- ✅ Multiple-comparison aware (DSR + BH FDR)
- ✅ Information-set leak-safe (Signal A/B distinct events)
- ✅ Transaction-cost robust (break-even 70+ bps/mo)
- ✅ Independent of FF5+UMD+LM (ablation)
- ✅ Honest about IS sector-artifact (Day 4 → Day 6 correction)
- ✅ Self-correcting in git log
- 🟡 OOS sample n=36 months (small CI; need more time pass)
- 🟡 Survivorship from PIT-union (acknowledged + partial mitigation)
- 🟡 Size proxy = price not market cap (free-data substitute)
- 🟡 Sample 935 → ~6000 pairs (well below 30K aspirational; pragmatic for free data)

**Verdict**: scout-grade-feasible *preliminary* candidate. Bottleneck is OOS power + WRDS access for proper market-cap matching. Methodology contribution (pre-reg + audit + matched + FDR-safe) is the actual headline regardless of sample size.

## Next 4 hours (auto)

Nothing user-facing changes until extraction finishes. Then:
1. Re-run `day4_run_all.py` + `day6_signal_matched.py` + `day7_*` + `day5_ablation_lm.py` with expanded sample.
2. Update paper_draft.md XX placeholders with final numbers.
3. Run final pytest sweep.
4. Final overnight summary commit.
