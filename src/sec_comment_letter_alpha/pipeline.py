"""Top-level orchestrator: fetch → parse → LLM features → factor → stats.

Day 1 entrypoints:
- `enqueue_universe(n)` — push N CIK requests to the SEC daemon queue
- `cache_status()` — count cached files in <coord_root>/sec-data/edgar-raw/upload-corresp/
- `dry_run(n)` — pull N cached filings, parse, LLM-extract, print JSON

Use via `python -m sec_comment_letter_alpha.pipeline <subcmd>`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from . import data_loader, features, parse, universe


def _coord_root() -> Path:
    # Load this repo's .env first (PORTFOLIO_COORD_ROOT), then the coord
    # repo's .env to pick up SEC_USER_AGENT and OPENROUTER_API_KEY.
    load_dotenv(Path.cwd() / ".env")
    coord = Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination"))
    load_dotenv(coord / ".env")
    return coord


def _cache_dir() -> Path:
    return _coord_root() / "sec-data" / "edgar-raw" / "upload-corresp"


def cache_status() -> dict:
    d = _cache_dir()
    files = list(d.glob("*.json")) if d.exists() else []
    n_miss = 0
    n_hit = 0
    n_filings = 0
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("miss"):
            n_miss += 1
        else:
            n_hit += 1
            n_filings += len(payload.get("filings") or [])
    return {
        "cache_dir": str(d),
        "files_total": len(files),
        "files_hit": n_hit,
        "files_miss": n_miss,
        "filings_total": n_filings,
    }


def enqueue_universe(n: int = 1000, *, seed: int = 17) -> dict:
    df = universe.load_universe()
    ciks = universe.sample_ciks(df, n=n, seed=seed)
    enqueued = 0
    already_cached = 0
    for c in ciks:
        lookup = data_loader.request_filings(c)
        if lookup.hit:
            already_cached += 1
        else:
            enqueued += 1
    return {
        "universe_size": len(df),
        "requested": len(ciks),
        "enqueued_new": enqueued,
        "already_cached": already_cached,
    }


def _select_uploads(n: int) -> list:
    """UPLOAD records from cache, HTML/TXT prioritised over PDF (daemon-cached
    PDFs are bytes-corrupted in the current pipeline; see parse.extract_pdf_text)."""
    all_uploads = [
        r
        for r in data_loader.iter_filings_in_cache(textual_only=True)
        if r.form == "UPLOAD"
    ]
    all_uploads.sort(key=lambda r: 1 if r.ext == ".pdf" else 0)
    return all_uploads[:n]


def dry_run(n: int = 10, *, model: str | None = None) -> list[dict]:
    """End-to-end on first N UPLOAD records in cache. Single-model."""
    _coord_root()  # populate env vars before client init
    from shared_utils.openrouter_client import OpenRouterClient

    records = _select_uploads(n)
    if not records:
        print("[dry-run] no UPLOAD filings in cache yet — daemon may still be fetching")
        return []

    client = OpenRouterClient(project=features.PROJECT_TAG)
    out: list[dict] = []
    for rec in records:
        segs = parse.split_into_segments(rec)
        if not segs:
            continue
        seg = segs[0]
        try:
            feat = features.extract_one(
                client,
                seg,
                model=model or "google/gemma-3-27b-it",
            )
            out.append(asdict(feat))
        except Exception as e:
            out.append({"error": str(e), "cik": rec.cik, "accession": rec.accession})
    return out


def ensemble_run(
    n: int = 10,
    *,
    models: tuple[str, ...] | None = None,
    prompt_version: str = features.DEFAULT_PROMPT_VERSION,
) -> dict:
    """Multi-model dry-run: each of N UPLOADs scored by every model in `models`.

    Returns a dict with `results` (per-segment per-model features) and
    `agreement` (pairwise Jaccard/Pearson/κ). `prompt_version` selects which
    prompt template (v1 / v2) is used; recorded in the output for provenance.
    """
    _coord_root()
    from shared_utils.openrouter_client import OpenRouterClient

    records = _select_uploads(n)
    if not records:
        print("[ensemble] no UPLOAD filings in cache yet")
        return {"results": [], "agreement": {}, "prompt_version": prompt_version}
    use_models = models or features.ENSEMBLE_DEFAULT

    client = OpenRouterClient(project=features.PROJECT_TAG)
    results: list[features.EnsembleResult] = []
    for rec in records:
        segs = parse.split_into_segments(rec)
        if not segs:
            continue
        results.append(features.extract_ensemble(
            client, segs[0], models=use_models, prompt_version=prompt_version,
        ))

    agreement = features.pairwise_agreement(results, models=use_models)
    serial_results = [
        {
            "cik": r.cik,
            "accession": r.accession,
            "date": r.date,
            "per_model": {m: asdict(f) for m, f in r.per_model.items()},
            "errors": r.errors,
        }
        for r in results
    ]
    serial_agreement = {f"{a}__VS__{b}": v for (a, b), v in agreement.items()}
    return {
        "prompt_version": prompt_version,
        "models": list(use_models),
        "results": serial_results,
        "agreement": serial_agreement,
    }


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sec_comment_letter_alpha.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_enq = sub.add_parser("enqueue", help="enqueue N CIK fetch requests")
    p_enq.add_argument("--n", type=int, default=1000)
    p_enq.add_argument("--seed", type=int, default=17)

    sub.add_parser("status", help="cache status counters")

    p_dry = sub.add_parser("dry-run", help="dry-run LLM extraction on first N cached filings")
    p_dry.add_argument("--n", type=int, default=10)
    p_dry.add_argument("--model", default=None)

    p_ens = sub.add_parser("ensemble", help="multi-model dry-run + pairwise agreement")
    p_ens.add_argument("--n", type=int, default=10)
    p_ens.add_argument("--models", nargs="*", default=None,
                       help="model slugs; default = features.ENSEMBLE_DEFAULT")
    p_ens.add_argument("--prompt-version", default=features.DEFAULT_PROMPT_VERSION,
                       choices=list(features.PROMPTS.keys()),
                       help="prompt template version (v1 = Day 1/2 baseline, v2 = anchored rubric)")

    args = p.parse_args(argv)
    if args.cmd == "enqueue":
        print(json.dumps(enqueue_universe(args.n, seed=args.seed), indent=2))
    elif args.cmd == "status":
        print(json.dumps(cache_status(), indent=2))
    elif args.cmd == "dry-run":
        out = dry_run(args.n, model=args.model)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "ensemble":
        models = tuple(args.models) if args.models else None
        out = ensemble_run(args.n, models=models, prompt_version=args.prompt_version)
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
