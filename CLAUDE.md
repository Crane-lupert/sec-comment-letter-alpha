# Project X — SEC Comment Letter LLM Alpha

## Purpose

SEC EDGAR UPLOAD (SEC → 기업) + CORRESP (기업 → SEC) 30K 건을 LLM ensemble 로 분석하여 **cross-sectional equity alpha** 도출. 채용용 포트폴리오 weapon.

**핵심 가설**: SEC comment 가 제기한 topic = firm-level 공시 취약점 사전 노출. Post-letter 60/90일 abnormal return 에 cross-sectional signal 존재.

**기여**: LLM 으로 comment letter 대화 → topic × severity × response pattern 구조화 후 cross-sectional factor 구성 및 FF5/momentum/PEAD/10-K sentiment 직교화. Prior-art 에서 직접 매칭 논문 찾지 못함.

---

## Dependencies (공유 인프라)

- SEC 데이터: `D:\vscode\portfolio-coordination\sec-data\edgar-raw\upload-corresp\` 에서 read. **직접 SEC 호출 금지**
- OpenRouter: `shared_utils.openrouter_client.OpenRouterClient(project="X")` 사용, project cap=8
- 체크포인트: 매일 08:00 / 20:00 KST `portfolio-coordination/checkpoints/<date>/project-x.md` 작성
- 상세 조율 규칙: `D:/vscode/portfolio-coordination/CLAUDE.md` 및 `D:/vscode/meta-harness/audits/2026-04-25-multi-repo-coordination.md`

---

## 실행 계획 + Advance Gate

### Day 1 — Pipeline + 수집 kickoff
작업:
- 환경 셋업 (`uv venv`, `uv pip install -e .`, `uv pip install -e D:/vscode/portfolio-coordination/shared-utils`, `.env` 작성)
- `shared_utils` import 확인 (`python -c "from shared_utils.sec_client import fetch_from_cache_or_queue"`)
- SEC queue 에 UPLOAD/CORRESP 요청 등록 (Russell 3000 × 2015-2024)
- `src/sec_comment_letter_alpha/pipeline.py` 골격 (data loader, segment extractor, LLM wrapper, statistical tests)
- Sanity 10 건 random sample 로 end-to-end dry-run
- 저녁 체크포인트 작성 (`shared_utils.checkpoint.write_checkpoint(project="X", ...)`)

**Advance Gate (Day 1 EOD)**:
- SEC queue 에 1000+ 요청 등록됨
- SEC agent 가 첫 100+ 건 fetch 완료 (공유 폴더에 존재)
- Pipeline dry-run 10 건 성공 (text → topic + severity JSON)

**미달 시**: Day 2 오전까지 infrastructure 디버그, 수집 목표 n=1500 으로 하향

### Day 2-3 — LLM feature extraction 확장
작업:
- 수집 완료분 (전 30K target) 의 LLM feature 배치 (Gemma + Llama + Claude ensemble)
- Feature schema: `{topic: Enum[~15], severity: 0-1, response_lag_days: int, sentiment: [-1,1], resolution: Enum}`
- Oracle validation: 사용자 수동 label 된 30 건 과 일치도 측정

**Advance Gate (Day 3 EOD)**:
- Feature extraction 전 sample 완료
- Oracle Cohen's κ > 0.7 (3 모델 간 일치도)

**미달 시**: prompt 재설계 Day 4 까지, feature schema 축소

### Day 4-5 — Cross-sectional factor 구성 + baseline
작업:
- Monthly rebalance factor: comment letter 받은 firm 의 severity-weighted short, control 로 non-letter firm long
- Baseline factor 재현:
  - FF5 + momentum + PEAD (Ball-Brown)
  - Full-transcript Loughran-McDonald sentiment (10-K)
- 직교화 regression, residual alpha 계산

**Advance Gate (Day 5 EOD)**:
- Main signal raw Sharpe 계산 완료
- Baseline 4종 재현 + OOS Sharpe 일치 (문헌 범위 ±30%)
- IS/OOS 분할: 2015-2021 IS, 2022-2024 OOS (frozen)

**미달 시**: baseline 재현 실패 시 pipeline bug 검색; 본 signal null 이면 methodology report 로 전환

### Day 6-7 — Rigor pass
작업:
- FDR 보정 (topic × severity × horizon = ~60 검정)
- Deflated Sharpe Ratio
- Bootstrap 신뢰구간 (B=1000, cluster by month)
- Sector/size/liquidity robustness
- Contamination audit (LLM 이 특정 기업 이름 기억 여부)

**Advance Gate (Day 7 EOD)**:
- 모든 rigor test 통과 OR 명시적 한계 기록
- Residual alpha CI 가 0 포함 여부 확인

**미달 시**: rigor 부족 부분 Week 2 초 보충

### Day 8-10 — Dashboard
작업:
- Streamlit app: topic × severity 히트맵, 선택 filter 시 cumulative return curve + CAR 분포
- Static screenshot fallback (Streamlit 실패 시)
- 배포: Streamlit Community Cloud 무료 tier 또는 GitHub Pages

**Advance Gate (Day 10 EOD)**: Dashboard URL 공개 + 5 use case 스크린샷

### Day 11-13 — Writeup + README
작업:
- 논문 draft 8-12p (SSRN 업로드 가능 수준): abstract / 방법론 / data / results / robustness / contribution
- GitHub README: 5분 skim flow, 리포 구조, 재현 instructions
- 인터뷰 demo script 5분

**Advance Gate (Day 13 EOD)**: 3 산출물 완성, 1 외부인이 README 읽고 프로젝트 이해 가능 판정

### Day 14 — Buffer
Final QA, CV/LinkedIn 반영.

---

## Data Pipeline Spec

### 수집
- Filing types: `UPLOAD`, `CORRESP`
- Universe: Russell 3000 (CIK list from Chen-Zimmermann accompanying universe data)
- Period: 2015-2024
- Target n: 30K correspondence pairs (comment + response)

### Parsing
- UPLOAD body → topic + severity (LLM)
- CORRESP body → response length, response lag, resolution indicator (LLM + 규칙)
- Pair matching: 같은 accession series 내 UPLOAD → CORRESP 매칭

### Feature Schema (LLM 출력 JSON)
```json
{
  "filing_date": "YYYY-MM-DD",
  "cik": "int",
  "topics": ["revenue_recognition", "segment_reporting", ...],  // multi-label
  "severity": 0.0-1.0,
  "response_lag_days": int,
  "resolution_signal": "accepted|partial|ongoing|unknown"
}
```

---

## 통계 Rigor Checklist

- [ ] IS/OOS 엄수 (OOS 는 분석 종료 직전까지 touch 금지)
- [ ] FDR 보정 (Benjamini-Hochberg)
- [ ] Deflated Sharpe Ratio (Bailey-Lopez de Prado)
- [ ] Bootstrap 신뢰구간 B=1000, cluster by month
- [ ] Newey-West 표준오차
- [ ] TC 모델 (10 bps fixed + 0.05 × ADV participation)
- [ ] Sector-neutral portfolio
- [ ] Contamination audit (LLM 이 firm name/date 을 이미 학습했는지)

---

## Abandon Criteria (사전 선언, 3+)

1. **Data infeasibility**: Day 2 EOD n < 1500 → Russell 3000 → S&P 500 축소
2. **Oracle κ < 0.5**: LLM feature 측정 noise 과다 → schema 단순화, 실패 지속 시 Day 5 까지 재설계
3. **Baseline 재현 실패**: PEAD 또는 LM sentiment 의 OOS Sharpe 가 문헌 범위 벗어남 → pipeline bug, 수정
4. **Budget overflow**: OpenRouter 누적 > $50 → 중단
5. **Main signal 정말로 null**: OOS residual Sharpe CI 가 0 포함 → **negative result 로 methodology 리포트** (논문 제목 바꿔 "Why SEC comment letter signals don't survive orthogonalization")

---

## Deliverables

- [ ] GitHub public repo
- [ ] Streamlit / static dashboard URL
- [ ] README (5분 skim)
- [ ] Paper draft 8-12p (SSRN 수준)
- [ ] Interview demo 5분 script
- [ ] CV/LinkedIn 1-line summary + 링크

---

## Interview Demo Script (5분)

1. (30s) "I built cross-sectional alt-data alpha from 30K SEC Comment Letter correspondence using multi-LLM ensemble. OOS Sharpe X, residual after FF5+momentum+PEAD+LM sentiment."
2. (90s) Dashboard 3-click: topic heatmap → pick 'revenue recognition' → 60-day CAR distribution
3. (90s) Methodology rigor: FDR, Deflated Sharpe, contamination audit, ablation vs LM sentiment
4. (60s) Novel contribution: no prior LLM alpha on SEC UPLOAD/CORRESP pair 의 구조화
5. (30s) "GitHub + paper + dashboard 모두 공개. 재현 1 command."
