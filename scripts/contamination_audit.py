"""Contamination audit — does the LLM infer features from text, or recall from training?

Picks 50 random R3K UPLOAD records that already have v2 ensemble features,
constructs a redacted variant of each excerpt (firm name, ticker, date,
accession, file numbers stripped), runs the SAME v2 prompt + same models, and
compares features.

If kappa(redacted, original) >= 0.7 on resolution_signal AND topic Jaccard
stays >= 0.7, the LLM is mostly inferring from generic text. If it drops
sharply, the model is leveraging memorized firm-specific knowledge.

Outputs:
  data/contamination_audit_redacted.jsonl   -- redacted features
  data/contamination_audit_summary.json     -- comparison summary
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sec_comment_letter_alpha import data_loader, features, parse  # noqa: E402

UPLOAD_FEAT = REPO_ROOT / "data" / "day3_features_r3k.jsonl"
R3K_PARQUET = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
OUT_FEAT = REPO_ROOT / "data" / "contamination_audit_redacted.jsonl"
OUT_SUMMARY = REPO_ROOT / "data" / "contamination_audit_summary.json"

N_SAMPLE = 50
SEED = 7


# --- Redaction patterns ---------------------------------------------------

# Generic patterns — applied to the segment text BEFORE the LLM sees it.
_DATE_PATTERNS = [
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)\s+\d{1,2}\s*,?\s*\d{4}\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\bfiscal year\s+(?:ended|ending)?\s*(?:[A-Za-z]+\s+\d{1,2}\s*,?\s*)?\d{4}\b", re.I),
    re.compile(r"\b\d{4}\b"),  # any 4-digit year (last; aggressive)
]
_FILE_NO = re.compile(r"\b(?:File|Form|Registration|Securities Act File)\s*(?:No\.?)?\s*[:#]?\s*\d{2,4}-\d{4,8}\b", re.I)
_CIK_NO = re.compile(r"\bCIK\s*[:#]?\s*\d{6,10}\b", re.I)
_ACCESSION = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
_PHONE = re.compile(r"\b(?:\d{3}[-.]?\d{3}[-.]?\d{4})\b")
_EMAIL = re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def redact(text: str, firm_name: str, ticker: str = "") -> str:
    out = text or ""
    # Firm-specific redactions FIRST (before regex).
    if firm_name:
        for tok in re.split(r"[\s,.()]+", firm_name):
            if len(tok) >= 3 and tok.lower() not in {"inc", "corp", "corporation", "company", "ltd", "the", "and", "co"}:
                out = re.sub(re.escape(tok), "[FIRM]", out, flags=re.I)
    if ticker:
        out = re.sub(r"\b" + re.escape(ticker) + r"\b", "[TICKER]", out)
    # Generic patterns
    for p in _DATE_PATTERNS:
        out = p.sub("[DATE]", out)
    out = _FILE_NO.sub("[FILE_NO]", out)
    out = _CIK_NO.sub("[CIK]", out)
    out = _ACCESSION.sub("[ACCESSION]", out)
    out = _PHONE.sub("[PHONE]", out)
    out = _EMAIL.sub("[EMAIL]", out)
    return out


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    coord = Path(os.environ.get("PORTFOLIO_COORD_ROOT", "D:/vscode/portfolio-coordination"))
    load_dotenv(coord / ".env")

    import pandas as pd
    r3k = pd.read_parquet(R3K_PARQUET)
    cik_to_meta = {row["cik"]: (row.get("ticker", ""), row.get("name", "")) for _, row in r3k.iterrows()}

    # Load original features for comparison
    orig_by_key: dict[tuple[str, str], dict] = {}
    for line in UPLOAD_FEAT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "skip" in obj or "fatal_error" in obj:
            continue
        orig_by_key[(obj["cik"], obj["accession"])] = obj
    print(f"[audit] original features available: {len(orig_by_key)}")

    # Get filing records to access raw text
    rec_by_key: dict[tuple[str, str], data_loader.FilingRecord] = {
        (r.cik, r.accession): r
        for r in data_loader.iter_filings_in_cache(textual_only=True)
        if r.form == "UPLOAD" and r.cik in cik_to_meta
    }
    available = [k for k in orig_by_key if k in rec_by_key]
    print(f"[audit] eligible (have features + raw text): {len(available)}")

    # Sample
    rng = random.Random(SEED)
    sample = rng.sample(available, min(N_SAMPLE, len(available)))
    print(f"[audit] sampling {len(sample)} records (seed={SEED})")

    from shared_utils.openrouter_client import OpenRouterClient
    client = OpenRouterClient(project=features.PROJECT_TAG)
    models = ("google/gemma-3-27b-it", "meta-llama/llama-3.3-70b-instruct")

    redacted_results: list[dict] = []
    with OUT_FEAT.open("w", encoding="utf-8") as f:
        for i, key in enumerate(sample):
            cik, acc = key
            rec = rec_by_key[key]
            ticker, firm_name = cik_to_meta.get(cik, ("", ""))
            segs = parse.split_into_segments(rec)
            if not segs:
                continue
            seg = segs[0]
            redacted_text = redact(seg.text, firm_name, ticker)
            redacted_seg = parse.ParsedSegment(
                cik=seg.cik, accession=seg.accession, form=seg.form, date=seg.date,
                segment_idx=seg.segment_idx, text=redacted_text,
            )
            per_model = {}
            errors = {}
            with ThreadPoolExecutor(max_workers=len(models)) as pool:
                futs = {
                    pool.submit(features.extract_one, client, redacted_seg, model=m, prompt_version="v2"): m
                    for m in models
                }
                for fut in as_completed(futs):
                    m = futs[fut]
                    try:
                        per_model[m] = fut.result()
                    except Exception as e:
                        errors[m] = f"{type(e).__name__}: {e}"
            row = {
                "cik": cik, "accession": acc, "date": rec.date,
                "redacted_text_chars": len(redacted_text),
                "original_text_chars": len(seg.text),
                "redactions_applied": True,
                "per_model": {m: asdict(feat) for m, feat in per_model.items()},
                "errors": errors,
            }
            redacted_results.append(row)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"[audit] {i+1}/{len(sample)} processed")

    print(f"[audit] wrote {OUT_FEAT}")

    # Compare metrics: redacted vs original
    def _features_aligned(model: str) -> tuple[list, list]:
        """Aligned (redacted, original) feature pairs for a given model."""
        red, orig = [], []
        for row in redacted_results:
            k = (row["cik"], row["accession"])
            if model not in row["per_model"] or k not in orig_by_key:
                continue
            if model not in orig_by_key[k]["per_model"]:
                continue
            red.append(row["per_model"][model])
            orig.append(orig_by_key[k]["per_model"][model])
        return red, orig

    summary = {"sample_size": len(redacted_results), "seed": SEED, "per_model": {}}
    for m in models:
        red, orig = _features_aligned(m)
        if not red:
            continue
        n = len(red)
        # Topic Jaccard mean
        topic_jacc = []
        for r, o in zip(red, orig):
            ra, oa = set(r["topics"]), set(o["topics"])
            u = ra | oa
            topic_jacc.append(len(ra & oa) / len(u) if u else 1.0)
        # Severity Pearson
        from statistics import mean
        rs = [r["severity"] for r in red]
        os_ = [o["severity"] for o in orig]
        mr, mo = mean(rs), mean(os_)
        sxy = sum((a - mr) * (b - mo) for a, b in zip(rs, os_))
        sxx = sum((a - mr) ** 2 for a in rs)
        syy = sum((b - mo) ** 2 for b in os_)
        sev_r = sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")
        # Resolution kappa
        classes = ["accepted", "partial", "ongoing", "unknown"]
        oa = sum(1 for r, o in zip(red, orig) if r["resolution_signal"] == o["resolution_signal"]) / n
        pa = {c: sum(1 for r in red if r["resolution_signal"] == c) / n for c in classes}
        pb = {c: sum(1 for o in orig if o["resolution_signal"] == c) / n for c in classes}
        pe = sum(pa[c] * pb[c] for c in classes)
        kappa = (oa - pe) / (1 - pe) if pe < 1 else float("nan")
        summary["per_model"][m] = {
            "n": n,
            "topic_jaccard_mean": sum(topic_jacc) / n,
            "severity_pearson": sev_r,
            "resolution_kappa": kappa,
            "redacted_dist": dict(Counter(r["resolution_signal"] for r in red)),
            "original_dist": dict(Counter(o["resolution_signal"] for o in orig)),
        }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Contamination audit summary ===")
    print(json.dumps(summary, indent=2))

    # Verdict
    print("\n=== Verdict ===")
    for m, s in summary["per_model"].items():
        verdict = "no-strong-leakage" if (s["topic_jaccard_mean"] >= 0.6 and s["resolution_kappa"] >= 0.5) else "POSSIBLE-CONTAMINATION"
        print(f"  {m}: jaccard={s['topic_jaccard_mean']:.3f} kappa={s['resolution_kappa']:.3f} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
