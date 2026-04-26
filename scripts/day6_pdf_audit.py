"""Day 6 — PDF extraction quality audit (heuristic, no SEC re-fetch).

Samples N random PDF UPLOADs from the daemon cache (R3K-only), runs
`parse.extract_pdf_text`, and scores each extraction with simple
text-quality heuristics:

  - n_words           : count of whitespace-delimited tokens
  - sentence_rate     : sentence-ending punctuation per 100 words
  - ws_ratio          : whitespace chars / total chars (proxy for bullet/garble noise)
  - alpha_ratio       : alphabetic chars / total chars (English text proxy)
  - printable_ratio   : printable chars / total chars (catches mojibake)
  - mean_token_len    : mean(len(token)) — garbled PDFs often have tons of 1-char tokens
  - confidence        : 0.0-1.0 weighted blend
  - score             : full | partial | garbled | empty

Outputs:
  data/day6_pdf_audit.json  — per-sample audit rows + summary
  docs/day6_pdf_audit.md    — methodology + summary + 5 highlighted samples

Run:
    .venv/Scripts/python.exe scripts/day6_pdf_audit.py --n 30
    .venv/Scripts/python.exe scripts/day6_pdf_audit.py --n 30 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sec_comment_letter_alpha import data_loader, parse  # noqa: E402

OUT_JSON = REPO_ROOT / "data" / "day6_pdf_audit.json"
OUT_MD = REPO_ROOT / "docs" / "day6_pdf_audit.md"
R3K_PARQUET = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"

PRINTABLE = set(string.printable)
SENTENCE_END = re.compile(r"[.!?]")
WS_RE = re.compile(r"\s")


def _quality_metrics(text: str) -> dict:
    """Return raw quality metrics for a single extracted string."""
    n_chars = len(text)
    if n_chars == 0:
        return {
            "n_chars": 0,
            "n_words": 0,
            "sentence_rate": 0.0,
            "ws_ratio": 0.0,
            "alpha_ratio": 0.0,
            "printable_ratio": 0.0,
            "mean_token_len": 0.0,
            "frac_short_tokens": 0.0,
        }
    tokens = text.split()
    n_words = len(tokens)
    sentence_count = len(SENTENCE_END.findall(text))
    ws_count = sum(1 for c in text if c.isspace())
    alpha_count = sum(1 for c in text if c.isalpha())
    printable_count = sum(1 for c in text if c in PRINTABLE)
    if n_words:
        mean_token_len = sum(len(t) for t in tokens) / n_words
        frac_short = sum(1 for t in tokens if len(t) <= 2) / n_words
    else:
        mean_token_len = 0.0
        frac_short = 0.0
    sentence_rate = (sentence_count / n_words * 100.0) if n_words else 0.0
    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "sentence_rate": round(sentence_rate, 3),
        "ws_ratio": round(ws_count / n_chars, 4),
        "alpha_ratio": round(alpha_count / n_chars, 4),
        "printable_ratio": round(printable_count / n_chars, 4),
        "mean_token_len": round(mean_token_len, 3),
        "frac_short_tokens": round(frac_short, 4),
    }


def _score(metrics: dict) -> tuple[str, float]:
    """Map quality metrics → (score_label, confidence_0_to_1).

    Heuristics:
      - empty     : n_words == 0
      - garbled   : n_words >= 1 but text fails English-readable thresholds
      - partial   : substantive but flagged on >=1 quality dim
      - full      : substantive AND clean on all dims

    Confidence blends the individual dims so a sample sitting near the
    boundary gets a midpoint score rather than a brittle 1/0 verdict.
    """
    if metrics["n_words"] == 0:
        return "empty", 0.0

    # Component scores, each in [0,1] -- 1 = looks good.
    # Words: ramp from 0 at 0 to 1 at 200+ words.
    s_words = min(1.0, metrics["n_words"] / 200.0)
    # Sentence rate: ideal 3-8 per 100 words; degrade outside.
    sr = metrics["sentence_rate"]
    if 3.0 <= sr <= 8.0:
        s_sent = 1.0
    elif 1.5 <= sr < 3.0 or 8.0 < sr <= 12.0:
        s_sent = 0.6
    elif 0.5 <= sr < 1.5 or 12.0 < sr <= 20.0:
        s_sent = 0.3
    else:
        s_sent = 0.0
    # Alpha ratio: English text typically 0.65-0.85; garbled <0.5 or >0.95.
    ar = metrics["alpha_ratio"]
    if 0.55 <= ar <= 0.90:
        s_alpha = 1.0
    elif 0.45 <= ar < 0.55 or 0.90 < ar <= 0.95:
        s_alpha = 0.5
    else:
        s_alpha = 0.0
    # Printable ratio: should be ~1 for clean text.
    pr = metrics["printable_ratio"]
    if pr >= 0.99:
        s_print = 1.0
    elif pr >= 0.95:
        s_print = 0.7
    elif pr >= 0.85:
        s_print = 0.3
    else:
        s_print = 0.0
    # Mean token len: English ~4-6; garbled either tiny or huge runs.
    mtl = metrics["mean_token_len"]
    if 3.5 <= mtl <= 7.0:
        s_mtl = 1.0
    elif 2.5 <= mtl < 3.5 or 7.0 < mtl <= 9.0:
        s_mtl = 0.6
    else:
        s_mtl = 0.2
    # Whitespace ratio: clean text ~0.10-0.22.
    wr = metrics["ws_ratio"]
    if 0.08 <= wr <= 0.25:
        s_ws = 1.0
    elif 0.05 <= wr < 0.08 or 0.25 < wr <= 0.40:
        s_ws = 0.5
    else:
        s_ws = 0.1

    # Weighted blend. Words drive volume, the rest drive readability.
    confidence = (
        0.20 * s_words
        + 0.20 * s_sent
        + 0.20 * s_alpha
        + 0.15 * s_print
        + 0.15 * s_mtl
        + 0.10 * s_ws
    )
    confidence = round(confidence, 4)

    # Categorical label
    if metrics["n_words"] < 50 and confidence < 0.45:
        label = "garbled"
    elif confidence >= 0.80 and metrics["n_words"] >= 200:
        label = "full"
    elif confidence >= 0.55:
        label = "partial"
    else:
        label = "garbled"
    return label, confidence


def _enumerate_pdf_uploads(allowed_ciks: set[str]) -> list[data_loader.FilingRecord]:
    out = []
    for r in data_loader.iter_filings_in_cache(textual_only=True):
        if r.form != "UPLOAD":
            continue
        if not r.primary.lower().endswith(".pdf"):
            continue
        if r.cik not in allowed_ciks:
            continue
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="number of PDF UPLOADs to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-highlights", type=int, default=5,
                   help="how many samples to include first-1000-char text in output")
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    args = p.parse_args(argv)

    if not R3K_PARQUET.exists():
        print(f"[day6_pdf_audit] FATAL: R3K parquet not found at {R3K_PARQUET}", file=sys.stderr)
        return 2
    allowed_ciks = set(pd.read_parquet(R3K_PARQUET)["cik"].astype(str))
    print(f"[day6_pdf_audit] R3K universe: {len(allowed_ciks)} CIKs")

    print("[day6_pdf_audit] enumerating PDF UPLOADs across cache (R3K-filtered) ...")
    pdf_uploads = _enumerate_pdf_uploads(allowed_ciks)
    print(f"[day6_pdf_audit] eligible PDF UPLOADs: {len(pdf_uploads)}")
    if len(pdf_uploads) < args.n:
        print(f"[day6_pdf_audit] FATAL: only {len(pdf_uploads)} eligible, need {args.n}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    sample = rng.sample(pdf_uploads, args.n)
    # Pick the highlights deterministically from the sample.
    highlight_idx = set(rng.sample(range(args.n), min(args.n_highlights, args.n)))

    rows = []
    for i, rec in enumerate(sample):
        text = parse.extract_pdf_text(rec.text)
        metrics = _quality_metrics(text)
        label, conf = _score(metrics)
        row = {
            "idx": i,
            "cik": rec.cik,
            "accession": rec.accession,
            "date": rec.date,
            "primary": rec.primary,
            "raw_blob_len": len(rec.text),
            **metrics,
            "confidence": conf,
            "score": label,
            "highlighted": i in highlight_idx,
        }
        if i in highlight_idx:
            row["sample_text_first_1000"] = text[:1000]
        rows.append(row)

    score_counter = Counter(r["score"] for r in rows)
    mean_conf = round(sum(r["confidence"] for r in rows) / len(rows), 4)
    by_label_conf = {
        lab: round(
            sum(r["confidence"] for r in rows if r["score"] == lab) / max(1, score_counter[lab]),
            4,
        )
        for lab in ("full", "partial", "garbled", "empty")
    }
    summary = {
        "n_sample": len(rows),
        "seed": args.seed,
        "n_highlights": args.n_highlights,
        "score_distribution": dict(score_counter),
        "score_pct": {k: round(100.0 * v / len(rows), 1) for k, v in score_counter.items()},
        "mean_confidence": mean_conf,
        "mean_confidence_by_label": by_label_conf,
        "n_words_quartiles": {
            "p25": int(pd.Series([r["n_words"] for r in rows]).quantile(0.25)),
            "p50": int(pd.Series([r["n_words"] for r in rows]).quantile(0.50)),
            "p75": int(pd.Series([r["n_words"] for r in rows]).quantile(0.75)),
            "max": int(max(r["n_words"] for r in rows)),
            "min": int(min(r["n_words"] for r in rows)),
        },
        "eligible_universe": len(pdf_uploads),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"[day6_pdf_audit] wrote {args.out_json}")

    _write_markdown(args.out_md, summary, rows)
    print(f"[day6_pdf_audit] wrote {args.out_md}")
    return 0


def _confidence_histogram(rows: list[dict], n_bins: int = 10) -> str:
    """ASCII histogram of confidence scores in [0,1]."""
    bins = [0] * n_bins
    for r in rows:
        c = max(0.0, min(0.999999, r["confidence"]))
        bins[int(c * n_bins)] += 1
    max_bin = max(bins) if any(bins) else 1
    lines = ["| bin | range | count | bar |", "|----|------|------|-----|"]
    for i, b in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bar = "#" * int(round(20 * b / max_bin)) if b else ""
        lines.append(f"| {i} | [{lo:.1f}, {hi:.1f}) | {b} | `{bar}` |")
    return "\n".join(lines)


def _write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    sd = summary["score_distribution"]
    sp = summary["score_pct"]
    mean_conf = summary["mean_confidence"]
    n = summary["n_sample"]
    seed = summary["seed"]
    eligible = summary["eligible_universe"]

    full_n = sd.get("full", 0)
    partial_n = sd.get("partial", 0)
    garbled_n = sd.get("garbled", 0)
    empty_n = sd.get("empty", 0)

    if (full_n + partial_n) / max(1, n) >= 0.8 and garbled_n / max(1, n) <= 0.10:
        verdict = "TRUSTABLE for downstream LLM (full+partial >= 80%, garbled <= 10%)."
    elif (full_n + partial_n) / max(1, n) >= 0.6:
        verdict = "CONDITIONAL — usable for LLM but flag low-confidence rows; consider drop-or-reextract policy."
    else:
        verdict = "NOT TRUSTABLE — extraction quality below threshold; revisit pypdf or PDF decoder."

    rows_sorted = sorted(rows, key=lambda r: r["confidence"])
    table_rows = ["| cik | accession | primary | n_words | sent_rate | ws_ratio | alpha | conf | score |",
                  "|------|-----------|---------|---------|-----------|----------|-------|------|-------|"]
    for r in rows_sorted:
        table_rows.append(
            f"| {r['cik']} | {r['accession']} | {r['primary']} | {r['n_words']} | "
            f"{r['sentence_rate']:.2f} | {r['ws_ratio']:.3f} | {r['alpha_ratio']:.3f} | "
            f"{r['confidence']:.3f} | {r['score']} |"
        )

    hist = _confidence_histogram(rows)

    highlights = [r for r in rows if r.get("highlighted")]
    highlight_blocks = []
    for h in highlights:
        snippet = h.get("sample_text_first_1000", "")
        snippet_md = snippet.replace("```", "``​`")
        highlight_blocks.append(
            f"### CIK {h['cik']} — accession {h['accession']}\n\n"
            f"- date: {h['date']}\n"
            f"- primary: {h['primary']}\n"
            f"- n_words: {h['n_words']} | sentence_rate: {h['sentence_rate']:.2f} | "
            f"alpha_ratio: {h['alpha_ratio']:.3f} | ws_ratio: {h['ws_ratio']:.3f}\n"
            f"- confidence: {h['confidence']:.3f} | score: **{h['score']}**\n\n"
            f"First 1000 chars (verbatim, for human eyeball):\n\n"
            f"```\n{snippet_md}\n```\n"
        )

    md = f"""# Day 6 — PDF Extraction Quality Audit

## Why

Roughly half of cached SEC UPLOAD/CORRESP filings arrive as PDFs. Post the
daemon-side latin-1 byte-preservation patch, `parse.extract_pdf_text`
returns non-empty strings on ~99% of PDFs, but "non-empty" does not imply
"semantically correct" — pypdf can mangle tables, footnotes, multi-column
layouts, and OCR-scanned letters. This audit measures actual extraction
quality on a random sample so we know whether to trust PDF text downstream
of the LLM feature pipeline.

## Method

1. Enumerated all cached UPLOAD filings whose `primary` ends in `.pdf`,
   restricted to the R3K universe (`data/universe_ciks_r3k.parquet`,
   {eligible} eligible PDF UPLOADs).
2. Drew a random sample of n={n} (seed={seed}).
3. For each, called `parse.extract_pdf_text(rec.text)` (the same code path
   used by `parse.split_into_segments`, no modifications), then computed
   six text-quality metrics:
   - **n_words** — substantive content proxy.
   - **sentence_rate** — sentence-ending punctuation per 100 words; English
     prose is typically 3-8 (Flesch / Hunt 1965 style).
   - **ws_ratio** — whitespace chars / total chars; clean prose is ~0.10-0.22.
   - **alpha_ratio** — alphabetic chars / total chars; clean English is
     ~0.65-0.85.
   - **printable_ratio** — printable chars / total; mojibake drags this
     below ~0.95.
   - **mean_token_len** — average token length in chars; garbled PDFs often
     show tons of 1-char tokens or massive run-on tokens.
4. Combined the six metrics into a `confidence ∈ [0,1]` weighted blend
   (see `_score` in `scripts/day6_pdf_audit.py`) and a categorical
   label `full | partial | garbled | empty`:
   - **empty**: 0 words extracted.
   - **garbled**: <50 words OR confidence <0.45.
   - **partial**: confidence in [0.55, 0.80).
   - **full**: confidence ≥0.80 and n_words ≥200.
5. No SEC re-fetch (project rule: never call SEC directly). The audit is
   self-consistent on cached extractions only.

## Results

### Distribution

| score | count | pct |
|-------|-------|-----|
| full     | {full_n} | {sp.get('full', 0.0):.1f}% |
| partial  | {partial_n} | {sp.get('partial', 0.0):.1f}% |
| garbled  | {garbled_n} | {sp.get('garbled', 0.0):.1f}% |
| empty    | {empty_n} | {sp.get('empty', 0.0):.1f}% |

- Mean confidence (all rows): **{mean_conf:.3f}**
- Mean confidence by label: full={summary['mean_confidence_by_label'].get('full', 0.0):.3f} | partial={summary['mean_confidence_by_label'].get('partial', 0.0):.3f} | garbled={summary['mean_confidence_by_label'].get('garbled', 0.0):.3f} | empty={summary['mean_confidence_by_label'].get('empty', 0.0):.3f}
- n_words quartiles: p25={summary['n_words_quartiles']['p25']} | p50={summary['n_words_quartiles']['p50']} | p75={summary['n_words_quartiles']['p75']} | min={summary['n_words_quartiles']['min']} | max={summary['n_words_quartiles']['max']}

### Confidence histogram

{hist}

### All sampled rows (sorted by confidence asc)

{chr(10).join(table_rows)}

## Highlighted samples (verbatim text — human eyeball check)

{chr(10).join(highlight_blocks)}

## Verdict

{verdict}

## Caveats

- Heuristic-only: no ground-truth comparison. A high-confidence row with
  pristine prose can still have wrong table values or mis-ordered
  multi-column content.
- The R3K filter is naive (no PIT membership), so a few sampled CIKs may
  not have been in R3K on the filing date. Acceptable for an extraction-
  quality audit.
- pypdf's `extract_text` is page-oriented; we lose page boundaries by
  joining with spaces. Downstream LLM is order-tolerant, so this is fine
  for topic/severity but would break for any positional analysis.

## Reproduce

```
.venv/Scripts/python.exe scripts/day6_pdf_audit.py --n 30 --seed {seed}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
