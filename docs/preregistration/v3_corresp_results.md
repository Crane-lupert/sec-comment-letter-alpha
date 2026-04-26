# v3-corresp held-out validation results

Counterpart to [`docs/preregistration_v3_corresp.md`](../preregistration_v3_corresp.md).
Schema was locked at git commit `23cabfe` BEFORE any v3-corresp extraction ran.

## Method (executed exactly as pre-registered)

1. Extract v3-corresp on the 100 train records ([`v3_corresp_split_train.json`](v3_corresp_split_train.json))
   → `data/day3_corresp_v3_train.jsonl`
2. Compute train κ + per-class distribution. (No schema edits permitted from this step on.)
3. Extract v3-corresp on the 1400 test records ([`v3_corresp_split_test.json`](v3_corresp_split_test.json))
   → `data/day3_corresp_v3_test.jsonl`
4. Report train κ vs test κ.

Models: `google/gemma-3-27b-it`, `meta-llama/llama-3.3-70b-instruct`. Same v2 ensemble pair as Day 3 baseline.

## Results

| set | n_shared | resolution_kappa | topic_jaccard | severity_pearson |
|---|---|---|---|---|
| TRAIN | 97 | 0.876 | 0.679 | 0.891 |
| TEST | 1400 | 0.856 | 0.736 | 0.888 |
| FULL (train+test) | 1497 | 0.857 | — | — |

**Validity check**: `|κ_train − κ_test| = 0.021 < 0.10` → schema is NOT data-fitted.

## Comparison: v2 vs v3-corresp

Same 1500 R3K CORRESP records, same models:

| metric | v2 (SEC-perspective enum) | v3-corresp (registrant POV) | Δ |
|---|---|---|---|
| resolution_kappa | 0.398 | 0.857 | +0.459 |
| topic_jaccard | 0.706 | 0.736 | +0.030 |
| severity_pearson | 0.896 | 0.888 | -0.008 |

The schema redesign moved κ from "fail" (below 0.7 gate) to "strong" (above 0.85). Topic and severity were unchanged because those fields didn't depend on the speech-act schema.

## Per-class distribution (TEST n=1400)

| class | gemma count | llama count |
|---|---|---|
| `agree_revise` | 707 | 634 |
| `explain_position` | 328 | 402 |
| `closing` | 191 | 144 |
| `supplemental` | 171 | 219 |
| `pushback` | 3 | 1 |

Both models cluster on the same dominant class (`agree_revise`, ~50%). `pushback` is rare (≤0.3%) but used — confirms the class is real (not dead weight) without driving κ.

## Internal-validity evidence (severity × intent crosstab on 867 Day 4 pairs)

`closing` intent uniquely associates with low UPLOAD severity:

| CORRESP intent | n | mean UPLOAD severity |
|---|---|---|
| supplemental | 97 | 0.543 |
| explain_position | 183 | 0.526 |
| agree_revise | 398 | 0.469 |
| disagree (gemma ≠ llama) | 113 | 0.411 |
| **closing** | **76** | **0.071** |

`closing` letters land on editorial-grade comments (severity 0.07) — substantive accounting issues (severity ≥ 0.4) never resolve in a single "no further action" letter. This is the economic structure we'd expect, and is captured by the schema without being explicitly designed for it.

## What this validates

1. **The redesign is signal, not fitting.** Train and test κ are within 0.02 of each other; if the schema had been over-fit to observed data, test κ would have collapsed.
2. **All 5 classes carry information.** No class is dead, none collapses to one another.
3. **Day 4 cross-section dataset has clean A+B features** ([data/day4_pairs.jsonl](../../data/day4_pairs.jsonl)): 867 R3K pairs with both LLM-inferred (A) and programmatic (B) features. 603 in the 2015-2024 OOS-friendly window.
