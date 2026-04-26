"""Day 3+ resumable CORRESP feature extraction.

Mirror of day3_extract.py for the CORRESP (registrant -> SEC) side. Same
models (gemma-3-27b + llama-3.3-70b), same v2 prompt — the schema is
transferable to registrant responses (they address the SAME topics, signal
their own resolution stance, and have severity proxies via topical depth).

Pairing with UPLOAD features (same review file) happens downstream in
parse.pair_upload_corresp / Day 4 cross-section construction. This script
just emits per-CORRESP features.

Output: data/day3_corresp_features.jsonl

Resumable identical to day3_extract.py (skip done (cik, accession) keys).

Run:
    .venv/Scripts/python.exe scripts/day3_corresp_extract.py --n 1500
    .venv/Scripts/python.exe scripts/day3_corresp_extract.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sec_comment_letter_alpha import data_loader, features, parse  # noqa: E402

OUTPUT = REPO_ROOT / "data" / "day3_corresp_features.jsonl"
DEFAULT_MODELS = ("google/gemma-3-27b-it", "meta-llama/llama-3.3-70b-instruct")
PROMPT_VERSION = "v2"
BUDGET_HARD_USD = 10.0
TARGET_FORM = "CORRESP"

SHUTDOWN = False


def _shutdown(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print(f"[day3-corresp] shutdown signal {signum} -- exiting after current record.")


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def _coord_root() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    coord = Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination"))
    load_dotenv(coord / ".env")
    return coord


def load_done_keys(path: Path) -> tuple[set[tuple[str, str]], int, int]:
    if not path.exists():
        return set(), 0, 0
    done = set()
    succ = err = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
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
        raw = p.read_text(encoding="utf-8")
        for end in range(len(raw), 0, -500):
            try:
                d = json.loads(raw[:end])
                return float(d["by_project"].get("X", 0))
            except Exception:
                continue
        return 0.0


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
                   help="parquet of CIK column to restrict universe")
    p.add_argument("--keys-filter", type=Path, default=None,
                   help="JSON list of [cik, accession] pairs to restrict to (held-out split)")
    p.add_argument("--record-parallelism", type=int, default=4,
                   help="how many records to process concurrently")
    args = p.parse_args(argv)

    if args.status:
        print(json.dumps(status(args.output), indent=2))
        return 0

    coord = _coord_root()
    from shared_utils.openrouter_client import OpenRouterClient  # noqa: E402

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done, succ_prev, err_prev = load_done_keys(args.output)
    print(f"[day3-corresp] resuming: {len(done)} done ({succ_prev} ok, {err_prev} skip/err)")

    spend_start = _read_x_spend(coord)
    print(f"[day3-corresp] X spend at start: ${spend_start:.4f} (cap ${BUDGET_HARD_USD})")
    if spend_start >= BUDGET_HARD_USD:
        print(f"[day3-corresp] FATAL: budget cap breached.")
        return 2

    universe = [
        r for r in data_loader.iter_filings_in_cache(textual_only=True) if r.form == TARGET_FORM
    ]
    if args.universe_filter:
        import pandas as pd
        allowed_ciks = set(pd.read_parquet(args.universe_filter)["cik"].astype(str))
        before = len(universe)
        universe = [r for r in universe if r.cik in allowed_ciks]
        print(f"[day3-corresp] universe filter applied: {before} -> {len(universe)} (allowed_ciks={len(allowed_ciks)})")

    if args.keys_filter:
        allowed_keys = {tuple(k) for k in json.loads(args.keys_filter.read_text(encoding="utf-8"))}
        before = len(universe)
        universe = [r for r in universe if (r.cik, r.accession) in allowed_keys]
        print(f"[day3-corresp] keys filter applied: {before} -> {len(universe)} (allowed_keys={len(allowed_keys)})")

    def _date_key(r):
        try:
            return -int(r.date.replace("-", "")[:8])
        except (AttributeError, ValueError):
            return 0
    universe.sort(key=_date_key)
    universe = universe[: args.n]
    todo = [r for r in universe if (r.cik, r.accession) not in done]
    print(f"[day3-corresp] universe(cap {args.n}): {len(universe)} | todo: {len(todo)} | models: {args.models}")
    if not todo:
        print("[day3-corresp] nothing to do.")
        return 0

    client = OpenRouterClient(project=features.PROJECT_TAG)
    state = {"processed": 0, "errors": 0, "next_check": 25}
    state_lock = threading.Lock()
    write_lock = threading.Lock()
    started = time.time()
    models = tuple(args.models)
    budget_breached = threading.Event()

    def _write(line: str):
        with write_lock:
            f.write(line)
            f.flush()

    def _process_one(rec):
        if SHUTDOWN or budget_breached.is_set():
            return
        segs = parse.split_into_segments(rec)
        if not segs:
            _write(json.dumps({
                "cik": rec.cik, "accession": rec.accession, "form": TARGET_FORM,
                "skip": "no_segments", "primary": rec.primary, "ext": rec.ext,
            }) + "\n")
            with state_lock:
                state["errors"] += 1
            return
        per_model = {}
        ensemble_errors = {}
        with ThreadPoolExecutor(max_workers=len(models)) as inner:
            futs = {
                inner.submit(
                    features.extract_one, client, segs[0],
                    model=m, prompt_version=args.prompt_version,
                ): m for m in models
            }
            for fut in as_completed(futs):
                m = futs[fut]
                try:
                    per_model[m] = fut.result()
                except Exception as e:  # noqa: BLE001
                    ensemble_errors[m] = f"{type(e).__name__}: {e}"
        if not per_model and ensemble_errors:
            _write(json.dumps({
                "cik": rec.cik, "accession": rec.accession, "form": TARGET_FORM,
                "fatal_error": f"all models failed: {ensemble_errors}",
            }) + "\n")
            with state_lock:
                state["errors"] += 1
            return
        _write(json.dumps({
            "cik": rec.cik, "accession": rec.accession, "date": rec.date,
            "form": TARGET_FORM, "primary": rec.primary, "ext": rec.ext,
            "per_model": {m: asdict(feat) for m, feat in per_model.items()},
            "errors": ensemble_errors,
            "prompt_version": args.prompt_version,
        }, default=str) + "\n")
        with state_lock:
            state["processed"] += 1
            if state["processed"] >= state["next_check"]:
                state["next_check"] += 25
                spend_now = _read_x_spend(coord)
                elapsed = time.time() - started
                rate = state["processed"] / elapsed if elapsed else 0
                eta = (len(todo) - state["processed"]) / rate if rate else float("inf")
                print(f"[day3-corresp] {state['processed']}/{len(todo)}, errs={state['errors']}, "
                      f"spend=${spend_now:.4f} (+${spend_now - spend_start:.4f}), "
                      f"rate={rate:.2f}/s, ETA={eta/60:.1f}min", flush=True)
                if spend_now >= BUDGET_HARD_USD:
                    print(f"[day3-corresp] BUDGET CAP HIT at ${spend_now:.4f}, stopping.", flush=True)
                    budget_breached.set()

    print(f"[day3-corresp] outer parallelism = {args.record_parallelism} records, inner = {len(models)} models", flush=True)
    with args.output.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.record_parallelism) as outer:
            list(outer.map(_process_one, todo))

    spend_end = _read_x_spend(coord)
    print(f"[day3-corresp] exit. processed_this_run={state['processed']} errors_this_run={state['errors']} "
          f"spend_delta=${spend_end - spend_start:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
