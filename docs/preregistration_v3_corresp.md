# Pre-registration: CORRESP schema v3-corresp + held-out validation

**Status**: Locked at git commit (this file). No edits to schema or split logic
permitted after this commit; deviation must be documented as a separate
amendment commit citing this file.

**Date**: 2026-04-26 KST (Day 3 evening)

## Why a redesign

The Day 2-3 ensemble (gemma-3-27b + llama-3.3-70b, prompt v2) produced
divergent classifications on CORRESP letters: resolution_signal Cohen's
κ = 0.398 (n=1500 R3K-only), well below the 0.7 advance gate. Diagnostic
distribution showed both models clustering on `partial` (gemma 66%, llama 64%)
because the v2 enum (`accepted | partial | ongoing | unknown`) was designed
from SEC's outbound speech-act perspective. CORRESP letters carry a
fundamentally different speech act — registrant's response — and the v2
enum forces a translation that each model performs differently.

Topic Jaccard (0.706) and severity Pearson (0.896) on the same v2 CORRESP
extraction were intact, confirming the failure is localized to one schema
field.

## Schema redesign principle

`response_intent` replaces `resolution_signal` for CORRESP only. Five
classes drawn from the registrant's speech act (NOT informed by inspecting
v2 outputs beyond the high-level "two-cluster on partial+unknown" finding):

| class | meaning | example anchor |
|---|---|---|
| `agree_revise` | will modify a future filing or filed a revised exhibit | "we will revise in our next amendment" |
| `explain_position` | defends current treatment, citing standards | "our current accounting is consistent with ASC..." |
| `supplemental` | provides additional info without changing | "in response to comment N, we supplementally advise..." |
| `pushback` | explicit disagreement with SEC | "we respectfully disagree" |
| `closing` | confirms all comments addressed, no further action | "we trust this is responsive" |

`unknown` removed; conservative default is `supplemental`.

## Held-out validation protocol

To test whether the redesign captures real signal vs fits the data we
already saw, the existing 1500 R3K CORRESP records are partitioned into:

- **train**: 100 records (deterministic random sample, seed=42)
- **test**: 1400 records (the rest)

**Sequence**:
1. Commit this document + schema (this commit).
2. Extract v3-corresp on the 100 train records.
3. Compute train κ and per-class distribution.
4. **Schema is locked** — no edits permitted, even if train κ disappoints.
5. Extract v3-corresp on the 1400 test records.
6. Report train κ and test κ side-by-side.

**Validity check**: |κ_train − κ_test| < 0.10 indicates no systematic
overfitting to train. Larger gap means schema redesign was data-fitted and
should be reported as such.

**Deliverables**:
- `data/day3_corresp_v3_train.jsonl` (100 records)
- `data/day3_corresp_v3_test.jsonl` (1400 records)
- `data/day3_corresp_v3_split.json` (the train/test key list, deterministic)

## Combined feature plan (Day 4)

Per UPLOAD↔CORRESP pair, the cross-section feature vector will combine:

**A — LLM-inferred (CORRESP-side from v3-corresp)**:
- `response_intent` ∈ 5 classes
- topic vector (gemma+llama agreement-weighted)
- severity (gemma+llama mean)

**B — programmatic (no LLM, deterministic)**:
- `response_lag_days` (already in `parse.pair_upload_corresp`)
- `response_length` (chars in CORRESP segment text)
- `n_segments` (numbered comment count in CORRESP)
- `topic_match` = Jaccard(UPLOAD topics, CORRESP topics)

Day 4 ablation: factor model on B-only vs B+A. If A adds incremental
out-of-sample alpha, retain. Otherwise drop A and report the failure.

## What this pre-registration does NOT permit

- Adjusting class anchors based on train results
- Adding/removing classes after step 1
- Re-defining the train/test split after seeing extractions
- Cherry-picking which models' outputs to keep

If any of those would help, that's evidence v3-corresp is fitted, and the
result should be reported under that limitation.
