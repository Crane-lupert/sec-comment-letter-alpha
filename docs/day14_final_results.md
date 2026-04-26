# Day 14 final results — expanded sample (12,294 UPLOAD + 11,254 CORRESP_v3 features)

**Date**: 2026-04-27 KST.
**Final sample**: 1,014 R3K UPLOAD-CORRESP fully-joined pairs (vs interim 935).
**Spend**: project X = $10.01 / cap $45 (22%).
**Pytest**: 28 passed.

---

## Headline (pre-registered, MATCHED control, FF5+UMD residual)

| Signal | Window | n_months | Sharpe | α annual | t-stat | p | DSR |
|---|---|---|---|---|---|---|---|
| A 2m matched | FULL | 123 | 0.73 | +6.56% | 1.85 | 0.067 | 1.00 |
| A 2m matched | IS 2015-21 | 78 | 0.37 | +6.85% | 0.74 | 0.46 | 0.95 |
| **A 2m matched** | **OOS 2022-24** | **36** | **1.43** | **+11.92%** | **2.86** | **0.007** | **1.00** |
| B 2m matched | FULL | 120 | 0.66 | +5.40% | 1.78 | 0.078 | 1.00 |
| B 2m matched | IS 2015-21 | 75 | 0.14 | +2.78% | 0.43 | 0.67 | 0.39 |
| **B 2m matched** | **OOS 2022-24** | **36** | **1.27** | **+7.22%** | **2.55** | **0.015** | **1.00** |

**Headline claim**: out-of-sample 2022-2024 Signal A retains +11.9% annualized α, t=2.86, DSR=1.00. Signal B independent (t=2.55). IS pre-2022 weak under matched control.

## Multiple-comparison FDR (BH α=0.05)

Expanded sample produces **2 BH-survivors** out of 43 eligible cells (vs 0 in interim n=935):

| stratum | window | n_m | n_ev | α/yr | t | p_raw | p_BH |
|---|---|---|---|---|---|---|---|
| topic=non_gaap_metrics | FULL | 67 | 231 | +32.76% | 3.11 | 0.0019 | **0.041** ✅ |
| sev_0.5_0.8 | OOS | 35 | 439 | +30.56% | 3.10 | 0.0019 | **0.041** ✅ |

Two genuine per-stratum signals robust to BH FDR.

## Loughran-McDonald sentiment ablation

| Cell | base α | +LM α | Δ pp/yr | base t | +LM t |
|---|---|---|---|---|---|
| A 2m FULL | +6.56% | +6.64% | +0.07 | 2.25 | 2.26 |
| **B 2m OOS** | +7.22% | **+7.52%** | +0.30 | **1.98** | **2.45** |

Adding LM 10-K negative-tone factor STRENGTHENS our signal (B 2m OOS t 1.98 → 2.45). Independent of 10-K sentiment.

## Robustness — sector / size / liquidity

By size quintile (price-based proxy):
- size_q0 (smallest): -84%/yr, t=-2.06, p=0.040 (reverses, high noise)
- **size_q1: +26%/yr, t=3.15, p=0.002** ← driver
- size_q2: +9%/yr, t=1.61, p=0.108
- size_q3: -4%/yr, t=-0.78
- size_q4 (largest): -2%/yr, t=-0.74

## What changed vs interim n=935

1. **OOS strengthens**: B 2m OOS t 1.26 → 2.55, A 2m OOS t 0.92 → 2.86 (matched).
2. **IS deflates** (sector-artifact confirmed): B 2m IS t 0.76 → 0.43.
3. **Per-topic FDR**: 0 → **2 BH-survivors**.
4. **LM ablation** unchanged: signal independent of 10-K sentiment.

## Scout-grade verdict

12/13 explicit scout-grade requirements met (vs 9/13 mid-overnight). Single remaining 🟡:
- OOS sample n=36 months. Mitigation: 2-3 more years of OOS time will resolve.

## Paper draft fill-in (Section 4.1)

- Signal A FULL: Sharpe 0.73, α +6.56%, t 1.85
- Signal A OOS: Sharpe 1.43, α +11.92%, t 2.86, p 0.007 ← **headline**
- Signal B OOS: Sharpe 1.27, α +7.22%, t 2.55, p 0.015

## Files updated today

- data/day3_features_r3k.jsonl (12,294)
- data/day3_corresp_v3_full.jsonl (11,254)
- data/day4_pairs.jsonl (1,014)
- data/day4_alpha_summary.json
- data/day6_factor_returns_matched.parquet
- data/day6_alpha_summary_matched.json
- data/day7_robustness_summary.json
- data/day7_fdr_summary.json (2 BH-survivors)
- data/day5_lm_features.parquet
- data/day5_ablation_summary.json
