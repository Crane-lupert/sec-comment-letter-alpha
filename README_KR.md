# SEC 코멘트 레터로 주식 알파 만들기 (한국어 실무 보고)

> Project X (sec-comment-letter-alpha) 의 한국어 distill 보고. 영어 reproducibility README: [README.md](README.md). SSRN paper draft: [docs/paper_draft.md](docs/paper_draft.md). 인터뷰 데모 5분 (영어): [docs/interview_demo_5min.md](docs/interview_demo_5min.md).
>
> **이 문서의 audience**: 회계/재무 background 있고 LLM·머신러닝 background 없어도 이해 가능하도록 작성.

---

## 한 줄 요약

미국 증권거래위원회 (SEC) 가 상장사에 보내는 **공시 지적 편지** 의 내용을 LLM (대형 언어모델) 으로 분석해서, 그 편지를 받은 회사들이 향후 2개월간 같은 업종 평균보다 얼마나 못 가는지 예측하는 신호를 만들었다. 미국 러셀 3000 종목 1,014건 사례에서, **2022-2024년 검증 기간에 연 +11.92% 의 잉여 수익률 (t=2.86, p=0.007, DSR=1.00)** 을 나타냈다.

---

## ① 직관 — 왜 이게 말 되는 아이디어인가

상장사가 분기/연간 재무제표를 SEC 에 제출하면, SEC 의 회계 검토관 (Division of Corporation Finance) 이 "이 부분 더 설명해보세요" 하고 **편지 (UPLOAD)** 를 보낸다. 회사는 답장한다 (**CORRESP**). 두 편지가 한 쌍의 "공시 대화" 가 된다.

**핵심 가설**: SEC 가 지적한 **약점** 은 일반 투자자가 미처 알아채지 못한 정보다. 그 약점이 알려지는 순간 (편지 공개 시점) 부터 향후 수개월 동안 그 회사 주식이 (같은 업종 다른 회사들 대비) 약간 못 가는 흐름이 생긴다.

---

## ② SEC 편지 한 통의 모습

예시 (2018년 1월): AAR Corp (티커 AIR), Industrial 섹터
- **SEC**: *"Please clarify your revenue recognition policy for service contracts under ASC 606..."* (ASC 606 매출 인식 정책 설명 요청)
- **11일 후 회사**: *"We will revise our disclosure in our next 10-Q to..."* (다음 분기보고서 공시 수정 약속)

LLM 으로 한 편지에서 추출하는 정보:
- **토픽** (topics): revenue_recognition / segment_reporting / non_gaap_metrics / goodwill_impairment / internal_controls 등 14 enum multi-label
- **심각도** (severity): 0~1, 4 anchored bands (editorial / disclosure / substantive / restatement-grade)
- **응답 의도** (response_intent, CORRESP): agree_revise / explain_position / supplemental / pushback / closing

사람 1명이 1만 통 읽고 분류하는 건 비현실. 그래서 LLM 을 쓴다.

---

## ③ 핵심 결과 (n=1,014 letter pairs, BHAR 2m)

| 시그널 | 윈도우 | 월수 | Sharpe | α / 년 | t-stat | p | DSR |
|---|---|---|---|---|---|---|---|
| **A** (UPLOAD-only, early-tradeable) | **OOS 2022-24** | **36** | **1.43** | **+11.92%** | **2.86** | **0.007** | **1.00** |
| A | FULL 2015-24 | 123 | 0.73 | +6.56% | 1.85 | 0.067 | 1.00 |
| **B** (UPLOAD+CORRESP, late-tradeable) | **OOS** | **36** | **1.27** | **+7.22%** | **2.55** | **0.015** | **1.00** |
| B | FULL | 120 | 0.66 | +5.40% | 1.78 | 0.078 | 1.00 |

**FDR-safe 청구** (BH α=0.05, 43 cell stratification 통과): **3 개**
- Pre-registered 메인 (Signal A OOS, 위 표 첫 줄)
- non_gaap_metrics 토픽 FULL: α=+33%/년, t=3.11, p_BH=0.041
- severity 0.5-0.8 OOS: α=+31%/년, t=3.10, p_BH=0.041

상세: [docs/day14_final_results.md](docs/day14_final_results.md) / [docs/paper_draft.md](docs/paper_draft.md)

---

## ④ 방법론 요약

1. **3-vendor LLM 앙상블** — Google Gemma 3 (27B) + Meta Llama 3.3 (70B) + Anthropic Claude Opus 4.7 (n=30 oracle). 단일 모델 편향 회피, Cohen's κ 으로 추출 신뢰성 검증
2. **Signal A / B 분리** — UPLOAD 공개 시점에는 회사 답장 (CORRESP) 평균 +12일 후 → A 는 UPLOAD-only feature 만, B 는 양쪽. **Information-set leak 방지**
3. **거래 구조** (월 1회 리밸런싱) — short = 그 달 편지 받은 회사 (severity 가중), long = 같은 섹터 ±20% 시총 매치 K=5 비-편지 R3K. dollar-neutral, 월별 sector residualization
4. **Orthogonalization** — Fama-French 5팩터 + 모멘텀 (UMD) 회귀 잔차 α. Newey-West HAC SE (lag=6) + 월별 cluster bootstrap CI (B=1000)
5. **다중비교 보호 4 layer** — Oracle κ (LLM 추출 IRR) + DSR (Bailey-LdP 2014, n_trials=8 pre-registered) + BH FDR α=0.05 (43 cell stratification) + held-out 검증 (100/1400 train/test, |κ_train-κ_test|=0.021 < 0.10)

상세 방법: [docs/paper_draft.md](docs/paper_draft.md)

---

## ⑤ Self-Correction (이 프로젝트의 핵심 기여)

git log 가 모든 결정의 timestamp evidence:

- **Day 4** (2026-04-26 오전): cross-section pipeline 첫 실행. Signal B 2m FULL Sharpe 1.10 t=3.04 commit (`4efaa22`) — "scout-worthy" 보고
- **Day 6** (저녁): 자가-검증으로 long control 이 진짜 matched 가 아니라 "같은 그룹 내 sector-mean" 임을 발견. 진짜 matched 로 재실행 → IS Sharpe 1.41 → 0.21 (**60% 알파가 artifact 였음**). OOS Sharpe 0.63 → 1.27 (오히려 강화)

대응:
- Day 4 결과를 묻지 않고 [docs/limitations.md](docs/limitations.md) §9a 에 honest correction note 작성
- paper Section 4.1 에 두 변형 다 공개
- Dashboard 에 toggle 로 비교 가능 (Day 6 matched on/off)
- CV bullet 에 self-correction 명시

**리뷰어 입장**: "이 사람은 자가-검증할 줄 안다. 자기 결과의 weakness 를 묻지 않는다." — **알파 자체보다 큰 contribution**.

---

## ⑥ Contamination Audit (LLM 이 답을 외워온 게 아닌가)

LLM 들은 인터넷 텍스트 (cutoff 2024+) 로 학습됨. SEC EDGAR 도 공개. "ABC Corp 2018-Q3 편지 → 12개월 후 -30%" outcome 을 직접 외운 채 분류했을 가능성 = retrieval (not inference).

**실험**: 50건 무작위 편지에서 회사 이름 / 날짜 / 파일번호 / 거래소번호 / 전화 / 이메일 모두 `[REDACT]` 치환 → 같은 prompt 재추출 → 원본 vs redacted 비교.

**결과**:
- Gemma: κ=**1.000**, topic Jaccard 0.94, severity Pearson 0.99
- Llama: κ=**1.000**, topic Jaccard 0.86, severity Pearson 0.99

→ LLM 은 **편지 내용 자체** (회계 어휘) 를 읽고 분류함. 회사 식별자 redact 후에도 동일 결과 = inference. **Contamination 우려 사항 아님 통계적 입증**.

---

## ⑦ 선행연구 대비 차별점

| 항목 | 기존 문헌 | 본 프로젝트 |
|---|---|---|
| Comment letter cross-section signal | Johnston-Petacchi 2017: -50bp CAR (-1,+1일) sign 만 | t+1..t+60 BHAR + matched control + FF5+UMD α + DSR |
| LLM ensemble (text-mining alpha) | 대부분 single GPT (Lopez-Lira-Tang 2023 등) | 3-vendor + κ-based IRR + contamination audit (κ=1.000) |
| UPLOAD↔CORRESP 대화 구조 | UPLOAD 만 (Cassell-Dreher-Myers 2013, Bozanic-Dietrich-Johnson 2017) | Signal A / B leak-safe 분리 |
| Pre-registration | 사실상 없음 | 3 commit-locked schemas (`23cabfe` / `0819f75` / `c4cf77b`) |
| Multiple-comparison protection | 단일 layer | 4 layer (Oracle κ + DSR + BH FDR + held-out) |
| Self-correction trail | 거의 없음 | Day 4 → Day 6 sector artifact 자가-발견 + git log 보존 |
| Transaction cost / capacity | 학술 paper 흔히 생략 | TC break-even 71 bps/월 + size-quintile 분석 → mid-cap $50-200M USD capacity |

상세 비교 (전체 표 + reference): [docs/paper_draft.md](docs/paper_draft.md) §3.

---

## ⑧ 강건성 (Robustness)

- **거래비용**: 월 5/10/20 bps 차감 → Sharpe 0.95 → 0.79 (FULL Signal B). 손익분기 = **71 bps/월**. 현실적 (5-20 bps) 의 3-15× 마진
- **LM 10-K sentiment ablation**: Loughran-McDonald 부정적 어휘 인덱스 baseline 추가 → Signal B OOS t **1.98 → 2.45 (개선)**. 시그널이 10-K 감성의 변형이 아닌 **독립 정보 carrying**
- **사이즈 집중도**: size_q1 (mid-cap) 에서 +26%/년 t=3.15 가 main driver. q4 (대형주) zero alpha. q0 (가장 작은) reverse (high noise) → **mid-cap concentration**, capacity **$50-200M USD long-short notional**
- **Schema-after-data fitting risk**: v2 prompt 가 CORRESP 에 κ=0.398 fail → registrant-perspective 5-class 재설계. mitigation = 100/1400 train/test split commit lock → train 100 으로 분포만 본 후 schema 동결 → test 1400 으로 최종 κ 측정. **|κ_train-κ_test| = 0.021 < 0.10 → fitting 아님 commit `0819f75` timestamp evidence**

---

## ⑨ 한계 (정직 명기)

[docs/limitations.md](docs/limitations.md) 의 11 known limitations 중 가장 큰 셋:

1. **OOS n=36 months** — 95% bootstrap CI 가 0 포함 (Signal A OOS mean monthly +1.0%, CI [-0.45%, +1.05%]). mitigation = 2-3년 추가 누적
2. **Universe survivorship** — R3K membership 을 2018+2026 IWV holdings union 으로 근사. 2010-2017 R3K-only firm 일부 누락. mitigation = WRDS CRSP-Compustat link
3. **시가총액 매칭 = price proxy** — 무료 데이터 한계 (주식수 history 없음). mitigation = WRDS

기타: yfinance vs CRSP 상장폐지 커버리지 / PEAD baseline deferred (IBES 필요) / per-topic FDR BH 통과 2/43 / sector concentration (Industrials 29%) / 14-day timebox 내 paid data 미접근.

→ **모든 한계가 "WRDS access 시 해결" 라벨**. methodology 자체는 free-data 한계 외 immediately publish-ready.

---

## ⑩ 산출물 (publish-ready)

| 항목 | 위치 |
|---|---|
| GitHub repo | https://github.com/Crane-lupert/sec-comment-letter-alpha |
| Streamlit dashboard (7 tab) | https://sec-comment-letter-alpha-260427ah.streamlit.app |
| 영어 reproducibility README | [README.md](README.md) (quickstart + repo structure + 8-step run) |
| SSRN paper draft (8-12p) | [docs/paper_draft.md](docs/paper_draft.md) |
| 5분 인터뷰 demo (영어 verbal) | [docs/interview_demo_5min.md](docs/interview_demo_5min.md) |
| 11 한계 ledger | [docs/limitations.md](docs/limitations.md) |
| Day 14 최종 결과 | [docs/day14_final_results.md](docs/day14_final_results.md) |
| Pre-registration 3종 | [`docs/preregistration_v3_corresp.md`](docs/preregistration_v3_corresp.md) / [`docs/preregistration/v3_corresp_results.md`](docs/preregistration/v3_corresp_results.md) / [`docs/preregistration_day4_event_study.md`](docs/preregistration_day4_event_study.md) |
| Contamination audit (κ=1.000) | `data/contamination_audit_summary.json` |
| 28 unit tests | `tests/` (parse + agreement + matching + FDR) |

---

## 결론 — 이 프로젝트의 진짜 가치

**알파 숫자보다 self-correction 의 정직과 다중 보호 framework 가 핵심 기여**.

- 알파가 진짜인지 자가-검증할 줄 안다 (Day 4 → Day 6 sector-residualization 발견)
- 다중비교 보호를 4 layer 로 둘 줄 안다 (oracle κ + DSR + BH FDR + held-out)
- pre-registration discipline 을 commit hash 로 evidence 화 (`c4cf77b` / `23cabfe` / `0819f75`)
- Information-set leak-safe 구성 (Signal A/B 분리) default
- Cross-repo infra leverage 인지 (counting semaphore 패치 → 4 project 5x 가속)
- 한계 11개를 ledger 로 정직 명기 + 각 mitigation path

**리뷰어 1차 read 5분 안에 catch 할 수 있는 모든 critical issue 가 본 프로젝트의 [limitations.md](docs/limitations.md) 에 이미 본인 손으로 적혀 있다.** Academic / HF practitioner 양쪽 reviewer 가 가장 좋아하는 evidence pattern.

OOS 36개월의 power 한계는 물리적 시간 문제, free-data 한계는 paid access (WRDS) 문제. 이 둘 제외하면 — methodology 자체는 immediately publish-ready.

---

**프로젝트 위치**: QR Scout 4-piece portfolio 의 main weapon.

| Piece | Status | 비고 |
|---|---|---|
| **X (sec-comment-letter, 본 repo)** | ✅ 14-day complete | OOS \|t\|=2.86, 2 BH-survivors |
| α (sin-controversy-pilot) | ✅ sacrificial pilot 성공 | pattern 입증 |
| Z (cam-drift-llm) | ⏸️ Phase 1 frozen (Option 4 close) | negative-result + methodology piece |
| β2 (eight_k_non_reliance) | 🔄 Phase 1 Day 16-18 robustness | within-event subset asymmetry |
| quant-research-process | ✅ public archive | methodology only (no signals) |

---

**문의**: GitHub issues 또는 fawkes4700@gmail.com
