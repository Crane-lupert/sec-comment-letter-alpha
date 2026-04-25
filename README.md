# sec-comment-letter-alpha (Project X)

**Scope**: 10-14 days | **Status**: scaffolding | **Target HF Pod**: Cross-section QR (Millennium, Thurn, Cubist, G-Research QA)

## What this is

Cross-sectional equity alpha from ~30K SEC `UPLOAD` (SEC → 기업) + `CORRESP` (기업 → SEC) correspondence via multi-LLM ensemble feature extraction. Extends SEC disclosure NLP literature; no direct LLM alpha match found on UPLOAD/CORRESP pairs.

Full execution plan + advance gates + abandon criteria: [`CLAUDE.md`](CLAUDE.md)

## Install

```powershell
# 1. 선행: portfolio-coordination 세팅 완료
cd D:/vscode/sec-comment-letter-alpha
uv venv
uv pip install -e .
uv pip install -e D:/vscode/portfolio-coordination/shared-utils
copy .env.example .env   # PORTFOLIO_COORD_ROOT 확인
```

## Day 1 시작

1. `D:/vscode/portfolio-coordination` 에서 `sec-agent-daemon.py` 가 실행 중인지 확인
2. 이 repo 에서 첫 작업: `src/sec_comment_letter_alpha/pipeline.py` 스캐폴드 + SEC queue 에 Russell 3000 × UPLOAD/CORRESP 요청 등록
3. 첫 20건 fetch 되는지 `sec-data/edgar-raw/upload-corresp/` 확인

## 규칙

- SEC 직접 호출 금지 — `shared_utils.sec_client` 만 사용
- OpenRouter 호출 = `OpenRouterClient(project="X")` 패턴
- 매일 08:00 / 20:00 KST 체크포인트 (`shared_utils.checkpoint.write_checkpoint`)

자세한 하네스: [`CLAUDE.md`](CLAUDE.md)
