"""Day 4 prep: join UPLOAD & CORRESP features into a per-pair dataset.

Combines:
  A — LLM-inferred features (from day3_features_r3k.jsonl + day3_corresp_v3_*.jsonl)
  B — programmatic features (response_lag, response_length, n_segments, topic_match)

Output: data/day4_pairs.jsonl
One row = one UPLOAD-CORRESP pair (R3K only, matched within 90 days, both sides
having LLM features).

Day 4 cross-section construction reads this file as ground truth.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sec_comment_letter_alpha import data_loader, parse  # noqa: E402

R3K_PARQUET = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
UPLOAD_FEATURES = REPO_ROOT / "data" / "day3_features_r3k.jsonl"
CORRESP_V2_FEATURES = REPO_ROOT / "data" / "day3_corresp_features_r3k.jsonl"
CORRESP_V3_TRAIN = REPO_ROOT / "data" / "day3_corresp_v3_train.jsonl"
CORRESP_V3_TEST = REPO_ROOT / "data" / "day3_corresp_v3_test.jsonl"
CORRESP_V3_FULL = REPO_ROOT / "data" / "day3_corresp_v3_full.jsonl"
OUT = REPO_ROOT / "data" / "day4_pairs.jsonl"


_NUMBERED = re.compile(r"^\s*(?:Comment\s*)?(\d{1,2})[.)\]]\s+", re.MULTILINE)


def _load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    """Index a JSONL file by (cik, accession). Skip rows with skip/fatal_error."""
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "skip" in obj or "fatal_error" in obj:
            continue
        out[(obj["cik"], obj["accession"])] = obj
    return out


def _consensus_topics(per_model: dict) -> tuple[list[str], int]:
    """Topics that BOTH models agree on (intersection)."""
    sets = [set(v["topics"]) for v in per_model.values() if v.get("topics")]
    if not sets:
        return [], 0
    inter = set.intersection(*sets)
    return sorted(inter), len(inter)


def _ensemble_severity(per_model: dict) -> float:
    sevs = [float(v["severity"]) for v in per_model.values() if v.get("severity") is not None]
    return mean(sevs) if sevs else 0.0


def _ensemble_categorical(per_model: dict) -> tuple[str, bool]:
    """Returns (chosen, agreed). If both models agree, that label; else 'disagree'."""
    cats = [v.get("resolution_signal") for v in per_model.values() if v.get("resolution_signal")]
    if not cats:
        return "unknown", False
    if all(c == cats[0] for c in cats):
        return cats[0], True
    return "disagree", False


def _segment_text(rec: data_loader.FilingRecord) -> str:
    segs = parse.split_into_segments(rec)
    return segs[0].text if segs else ""


def _n_numbered_comments(text: str) -> int:
    return len(_NUMBERED.findall(text or ""))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


@dataclass
class PairRow:
    pair_id: str
    cik: str
    ticker: str
    sector: str
    upload_date: str
    upload_accession: str
    corresp_date: str
    corresp_accession: str
    # A — UPLOAD ensemble (gemma+llama, v2 prompt)
    upload_topics_consensus: list[str]
    upload_topics_n_consensus: int
    upload_severity_mean: float
    upload_resolution_signal: str
    upload_resolution_agreed: bool
    # A — CORRESP ensemble (gemma+llama, v2 prompt -- v2 fallback for severity/topics)
    corresp_topics_consensus: list[str]
    corresp_severity_mean: float
    # A — CORRESP v3-corresp (response_intent if available, else None)
    corresp_response_intent: str | None
    corresp_response_intent_agreed: bool | None
    # B — programmatic
    response_lag_days: int
    response_length_chars: int
    n_segments_corresp: int
    topic_match_jaccard: float


def main() -> int:
    r3k = pd.read_parquet(R3K_PARQUET)
    cik_meta = {row["cik"]: row for _, row in r3k.iterrows()}
    print(f"[day4] R3K universe: {len(cik_meta)} CIKs")

    up_feat = _load_jsonl(UPLOAD_FEATURES)
    co_feat = _load_jsonl(CORRESP_V2_FEATURES)
    # Merge v3-corresp from train + test + full (later sources win on key collision)
    v3_feat = {
        **_load_jsonl(CORRESP_V3_TRAIN),
        **_load_jsonl(CORRESP_V3_TEST),
        **_load_jsonl(CORRESP_V3_FULL),
    }
    print(f"[day4] UPLOAD features: {len(up_feat)}")
    print(f"[day4] CORRESP v2 features: {len(co_feat)}")
    print(f"[day4] CORRESP v3-corresp features: {len(v3_feat)}")

    # Pull all R3K filings, build text lookup keyed by (cik, accession)
    text_lookup: dict[tuple[str, str], data_loader.FilingRecord] = {}
    for rec in data_loader.iter_filings_in_cache(textual_only=True):
        if rec.cik in cik_meta:
            text_lookup[(rec.cik, rec.accession)] = rec
    print(f"[day4] R3K filing text records loaded: {len(text_lookup)}")

    # Pair via parse.pair_upload_corresp on R3K records only
    r3k_records = [r for r in text_lookup.values()]
    pairs = parse.pair_upload_corresp(r3k_records, window_days=90)
    matched = [(u, c) for u, c in pairs if c is not None]
    print(f"[day4] R3K UPLOAD-CORRESP pairs: total_uploads={len(pairs)}, matched={len(matched)}")

    rows: list[PairRow] = []
    drop_reasons = Counter()
    for upload, corresp in matched:
        u_key = (upload.cik, upload.accession)
        c_key = (corresp.cik, corresp.accession)

        # Need LLM features on BOTH sides to be a Day 4 row
        if u_key not in up_feat:
            drop_reasons["no_upload_features"] += 1
            continue
        if c_key not in co_feat:
            drop_reasons["no_corresp_v2_features"] += 1
            continue

        u_pm = up_feat[u_key]["per_model"]
        c_pm = co_feat[c_key]["per_model"]

        # A: UPLOAD
        u_topics_cons, u_topics_n = _consensus_topics(u_pm)
        u_sev = _ensemble_severity(u_pm)
        u_res, u_res_ok = _ensemble_categorical(u_pm)

        # A: CORRESP v2 (topics + severity)
        c_topics_cons, _ = _consensus_topics(c_pm)
        c_sev = _ensemble_severity(c_pm)

        # A: CORRESP v3-corresp (response_intent) -- optional
        v3 = v3_feat.get(c_key)
        if v3:
            v3_pm = v3["per_model"]
            c_intent, c_intent_ok = _ensemble_categorical(v3_pm)
        else:
            c_intent, c_intent_ok = None, None

        # B: programmatic
        c_text = _segment_text(corresp)
        lag = (parse._parse_date(corresp.date) - parse._parse_date(upload.date)).days

        meta = cik_meta[upload.cik]
        rows.append(PairRow(
            pair_id=f"{upload.cik}_{upload.accession}_{corresp.accession}",
            cik=upload.cik,
            ticker=str(meta.get("ticker", "")),
            sector=str(meta.get("sector", "")),
            upload_date=upload.date,
            upload_accession=upload.accession,
            corresp_date=corresp.date,
            corresp_accession=corresp.accession,
            upload_topics_consensus=u_topics_cons,
            upload_topics_n_consensus=u_topics_n,
            upload_severity_mean=u_sev,
            upload_resolution_signal=u_res,
            upload_resolution_agreed=u_res_ok,
            corresp_topics_consensus=c_topics_cons,
            corresp_severity_mean=c_sev,
            corresp_response_intent=c_intent,
            corresp_response_intent_agreed=c_intent_ok,
            response_lag_days=lag,
            response_length_chars=len(c_text),
            n_segments_corresp=_n_numbered_comments(c_text),
            topic_match_jaccard=_jaccard(u_topics_cons, c_topics_cons),
        ))

    print(f"[day4] kept {len(rows)} / {len(matched)} pairs (dropped: {dict(drop_reasons)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), default=str) + "\n")
    print(f"[day4] wrote {OUT}")

    # Summary stats
    if rows:
        years = Counter(r.upload_date[:4] for r in rows if r.upload_date)
        in_target = sum(n for y, n in years.items() if "2015" <= y <= "2024")
        print(f"[day4] year span: {min(years)}..{max(years)}, in 2015-2024: {in_target}")

        intents = Counter(r.corresp_response_intent for r in rows if r.corresp_response_intent)
        print(f"[day4] v3-corresp intent coverage: {sum(intents.values())} rows have v3 features")
        print(f"[day4] intent dist: {dict(intents)}")

        lags = sorted(r.response_lag_days for r in rows)
        print(f"[day4] response_lag_days: median={lags[len(lags)//2]}, p90={lags[int(len(lags)*0.9)]}, max={lags[-1]}")

        print(f"[day4] topic_match: mean={mean(r.topic_match_jaccard for r in rows):.3f}")
        print(f"[day4] response_length: median={sorted(r.response_length_chars for r in rows)[len(rows)//2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
