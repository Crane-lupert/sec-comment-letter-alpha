"""Day-1 EOD checkpoint emitter — call from CLI at 20:00 KST.

Reads live state from the coordination repo and produces a concrete advance-gate
verdict. Idempotent across the same KST date (write_checkpoint appends).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sec_comment_letter_alpha.pipeline import cache_status  # noqa: E402
from shared_utils.checkpoint import write_checkpoint  # noqa: E402


def _coord_root() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    return Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination"))


def _queue_depth() -> int:
    return len(list((_coord_root() / "sec-data" / "queue").glob("*.json")))


def _gate_verdict(stat: dict, queue_depth: int, dry_run_ok: int) -> tuple[str, list[str]]:
    parts: list[str] = []
    enq_total = stat["files_total"] + queue_depth  # cached + still-queued = total registered
    parts.append(f"queue 등록 total={enq_total} (cached={stat['files_total']}, queued={queue_depth})")
    parts.append(f"cache hit files={stat['files_hit']}, miss={stat['files_miss']}, filings_total={stat['filings_total']}")
    parts.append(f"dry-run JSON OK n={dry_run_ok}")

    pass_enq = enq_total >= 1000
    pass_cache = stat["files_hit"] >= 100
    pass_dry = dry_run_ok >= 10

    if pass_enq and pass_cache and pass_dry:
        verdict = "PASS — Day 1 advance gate 모두 통과"
    elif pass_enq and not pass_cache:
        verdict = "PARTIAL — enqueue 통과, fetch 미완 (daemon 처리 진행 중). Day 2 오전까지 infra 디버그 후 재평가."
    elif pass_enq and pass_cache and not pass_dry:
        verdict = "PARTIAL — fetch 충분, dry-run 미달. LLM/parse 디버그 필요."
    else:
        verdict = "FAIL — enqueue 미달. 미달 조치: Russell 3000 리스트 재확인."
    return verdict, parts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run-ok", type=int, default=0,
                   help="number of dry-run filings whose LLM JSON parsed successfully")
    p.add_argument("--openrouter-usd", type=float, default=0.0)
    p.add_argument("--blockers", nargs="*", default=[])
    p.add_argument("--next-plan", nargs="*", default=[])
    p.add_argument("--cross-project-notes", default="")
    args = p.parse_args(argv)

    stat = cache_status()
    qdepth = _queue_depth()
    verdict, detail = _gate_verdict(stat, qdepth, args.dry_run_ok)

    progress = [
        "환경 셋업 완료 (uv venv + sec-comment-letter-alpha + shared-utils editable install)",
        "shared_utils import sanity OK (sec_client / openrouter_client / checkpoint)",
        "Pipeline 스캐폴드 작성: data_loader / parse / features / stats / universe / pipeline (모듈별 docstring + skeleton)",
        "Universe 부트스트랩: SEC company_tickers.json one-shot → data/universe_ciks.parquet (1500 CIKs)",
        f"SEC queue enqueue: 1000 CIK × upload-corresp (project=X)",
        *detail,
    ]
    blockers = list(args.blockers)
    next_plan = list(args.next_plan) or [
        "Day 2 오전: daemon fetch 진행 모니터링, 100+ 도달 시 dry-run 즉시 실행",
        "Day 2 오후: LLM ensemble (gemma-2-9b + llama-3.3-70b + claude-3.5-sonnet) feature schema 확정",
        "Oracle 30건 수동 라벨 시드 작성 (κ 측정 준비)",
    ]

    path = write_checkpoint(
        project="X",
        progress=progress,
        blockers=blockers,
        next_plan=next_plan,
        advance_gate_status=verdict,
        budget={
            "openrouter_usd": args.openrouter_usd,
            "openrouter_soft_cap": 45,
            "data_n_enqueued": stat["files_total"] + qdepth,
            "data_n_filings_cached": stat["filings_total"],
        },
        cross_project_notes=args.cross_project_notes,
    )
    print(f"[checkpoint] wrote {path}")
    print(json.dumps({"verdict": verdict, "stat": stat, "queue_depth": qdepth, "dry_run_ok": args.dry_run_ok}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
