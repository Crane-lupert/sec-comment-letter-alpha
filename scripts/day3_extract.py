"""Day 3 resumable ensemble feature extraction.

Runs `gemma-3-27b-it` + `llama-3.3-70b-instruct` (v2 prompt) on every textual
UPLOAD in the cache, capped at `--n`. Opus oracle on the first 30 records is
already in `data/day2_ensemble_v2_opus.json` -- this script does NOT re-run it.

Output: `data/day3_features.jsonl` (one JSON object per record, append-only).

Resumability:
  - On startup, read existing JSONL and build a set of done (cik, accession) keys.
  - Skip records whose key is already in the set.
  - Each successful record is flushed to disk before moving on.
  - A truncated tail line from a crash is silently skipped on re-read.

Run:
    .venv/Scripts/python.exe scripts/day3_extract.py --n 1500
    .venv/Scripts/python.exe scripts/day3_extract.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sec_comment_letter_alpha import data_loader, features, parse  # noqa: E402

OUTPUT = REPO_ROOT / "data" / "day3_features.jsonl"
DEFAULT_MODELS = ("google/gemma-3-27b-it", "meta-llama/llama-3.3-70b-instruct")
PROMPT_VERSION = "v2"
BUDGET_HARD_USD = 10.0  # X total spend ceiling for this run

SHUTDOWN = False


def _shutdown(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print(f"[day3] shutdown signal {signum} -- finishing current record then exiting.")


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def _coord_root() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    coord = Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination"))
    load_dotenv(coord / ".env")
    return coord


def load_done_keys(path: Path) -> tuple[set[tuple[str, str]], int, int]:
    """Return (done_keys, success_count, error_count)."""
    if not path.exists():
        return set(), 0, 0
    done: set[tuple[str, str]] = set()
    succ = 0
    err = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue  # truncated tail line
        key = (obj.get("cik", ""), obj.get("accession", ""))
        if not key[0] or not key[1]:
            continue
        done.add(key)
        if "fatal_error" in obj or "skip" in obj:
            err += 1
        else:
            succ += 1
    return done, succ, err


def _read_x_spend(coord_root: Path) -> float:
    p = coord_root / "openrouter-usage.json"
    try:
        return float(json.loads(p.read_text(encoding="utf-8"))["by_project"].get("X", 0))
    except Exception:
        # concurrent-write race: scan for the largest valid prefix
        raw = p.read_text(encoding="utf-8")
        for end in range(len(raw), 0, -500):
            try:
                d = json.loads(raw[:end])
                return float(d["by_project"].get("X", 0))
            except Exception:
                continue
        return 0.0  # unparseable; let the run proceed (LLM call will surface real errors)


def status(path: Path = OUTPUT) -> dict:
    done, succ, err = load_done_keys(path)
    return {"output": str(path), "done_keys": len(done), "success": succ, "errors_or_skips": err}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--prompt-version", default=PROMPT_VERSION,
                   choices=list(features.PROMPTS.keys()))
    p.add_argument("--status", action="store_true")
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--universe-filter", type=Path, default=None,
                   help="parquet of CIK column to restrict universe (e.g. data/universe_ciks_r3k.parquet)")
    args = p.parse_args(argv)

    if args.status:
        print(json.dumps(status(args.output), indent=2))
        return 0

    coord = _coord_root()
    from shared_utils.openrouter_client import OpenRouterClient  # noqa: E402

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done, succ_prev, err_prev = load_done_keys(args.output)
    print(f"[day3] resuming: {len(done)} already-done ({succ_prev} ok, {err_prev} skipped/errored)")

    spend_start = _read_x_spend(coord)
    print(f"[day3] X spend at start: ${spend_start:.4f} (hard cap ${BUDGET_HARD_USD})")
    if spend_start >= BUDGET_HARD_USD:
        print(f"[day3] FATAL: budget cap already breached, refusing to run.")
        return 2

    # Universe: textual UPLOADs, sorted to put non-PDF first.
    universe = [
        r for r in data_loader.iter_filings_in_cache(textual_only=True) if r.form == "UPLOAD"
    ]

    # Optional CIK filter (e.g. real Russell 3000 membership)
    if args.universe_filter:
        import pandas as pd  # local import to avoid hard dep if unused
        allowed_ciks = set(pd.read_parquet(args.universe_filter)["cik"].astype(str))
        before = len(universe)
        universe = [r for r in universe if r.cik in allowed_ciks]
        print(f"[day3] universe filter applied: {before} -> {len(universe)} records (allowed_ciks={len(allowed_ciks)})")

    # Sort by date DESC -- prioritize 2015-2024 target-window records.
    # PDF/non-PDF distinction is moot post-daemon-patch (PDFs extract cleanly).
    def _date_key(r):
        try:
            return -int(r.date.replace("-", "")[:8])
        except (AttributeError, ValueError):
            return 0
    universe.sort(key=_date_key)
    universe = universe[: args.n]

    todo = [r for r in universe if (r.cik, r.accession) not in done]
    print(f"[day3] universe (cap {args.n}): {len(universe)} | to-do: {len(todo)} | models: {args.models}")

    if not todo:
        print("[day3] nothing to do -- all records already processed.")
        return 0

    client = OpenRouterClient(project=features.PROJECT_TAG)
    processed = 0
    errors_run = 0
    started = time.time()
    next_budget_check_at = 25
    models = tuple(args.models)

    with args.output.open("a", encoding="utf-8") as f:
        for rec in todo:
            if SHUTDOWN:
                print("[day3] shutdown requested, breaking loop")
                break

            segs = parse.split_into_segments(rec)
            if not segs:
                f.write(json.dumps({
                    "cik": rec.cik, "accession": rec.accession,
                    "skip": "no_segments", "primary": rec.primary, "ext": rec.ext,
                }) + "\n")
                f.flush()
                errors_run += 1
                continue

            # Per-model parallelism: each model has its own file-lock semaphore,
            # so threads on different models run concurrently, halving wall time.
            per_model = {}
            ensemble_errors = {}
            with ThreadPoolExecutor(max_workers=len(models)) as pool:
                futs = {
                    pool.submit(
                        features.extract_one, client, segs[0],
                        model=m, prompt_version=args.prompt_version,
                    ): m
                    for m in models
                }
                for fut in as_completed(futs):
                    m = futs[fut]
                    try:
                        per_model[m] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        ensemble_errors[m] = f"{type(e).__name__}: {e}"

            if not per_model and ensemble_errors:
                f.write(json.dumps({
                    "cik": rec.cik, "accession": rec.accession,
                    "fatal_error": f"all models failed: {ensemble_errors}",
                }) + "\n")
                f.flush()
                errors_run += 1
                continue

            result = features.EnsembleResult(
                cik=segs[0].cik, accession=segs[0].accession, date=segs[0].date,
                per_model=per_model, errors=ensemble_errors,
            )

            f.write(json.dumps({
                "cik": result.cik,
                "accession": result.accession,
                "date": result.date,
                "primary": rec.primary,
                "ext": rec.ext,
                "per_model": {m: asdict(feat) for m, feat in result.per_model.items()},
                "errors": result.errors,
                "prompt_version": args.prompt_version,
            }, default=str) + "\n")
            f.flush()
            processed += 1

            if processed >= next_budget_check_at:
                next_budget_check_at += 25
                spend_now = _read_x_spend(coord)
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed else 0
                eta_sec = (len(todo) - processed) / rate if rate else float("inf")
                print(f"[day3] {processed}/{len(todo)} processed, "
                      f"errs={errors_run}, spend=${spend_now:.4f} (Δ ${spend_now - spend_start:.4f}), "
                      f"rate={rate:.2f}/s, ETA={eta_sec/60:.1f}min")
                if spend_now >= BUDGET_HARD_USD:
                    print(f"[day3] BUDGET CAP HIT: spend ${spend_now:.4f} >= ${BUDGET_HARD_USD}, exiting.")
                    break

    spend_end = _read_x_spend(coord)
    print(f"[day3] exit. processed_this_run={processed} errors_this_run={errors_run} "
          f"spend_delta=${spend_end - spend_start:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
