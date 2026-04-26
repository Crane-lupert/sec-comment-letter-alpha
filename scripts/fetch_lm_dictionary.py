"""Download Loughran-McDonald sentiment master dictionary from Notre Dame SRAF.

Output: data/lm_master_dictionary.csv  (the canonical CSV)
        data/lm_negative_words.txt     (one word per line, the negative list)
        data/lm_positive_words.txt
        data/lm_uncertainty_words.txt
        data/lm_litigious_words.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "data" / "lm_master_dictionary.csv"

URL_CANDIDATES = [
    "https://sraf.nd.edu/textual-analysis/wp-content/uploads/sites/3/2024/02/Loughran-McDonald_MasterDictionary_1993-2023.csv",
    "https://sraf.nd.edu/textual-analysis/wp-content/uploads/sites/3/2023/01/Loughran-McDonald_MasterDictionary_1993-2022.csv",
    "https://sraf.nd.edu/wp-content/uploads/sites/3/2024/02/Loughran-McDonald_MasterDictionary_1993-2023.csv",
]


def fetch() -> bytes:
    last_err = None
    for u in URL_CANDIDATES:
        try:
            r = httpx.get(u, headers={"User-Agent": "Mozilla/5.0 research"},
                          follow_redirects=True, timeout=60)
            if r.status_code == 200 and len(r.content) > 100_000:
                print(f"[lm] downloaded from {u}")
                return r.content
            print(f"[lm] {u}: status={r.status_code} len={len(r.content)}, trying next")
        except Exception as e:
            last_err = e
            print(f"[lm] {u}: err {type(e).__name__}: {e}")
    raise RuntimeError(f"all LM dictionary URLs failed; last={last_err}")


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch()
    OUT_CSV.write_bytes(raw)
    df = pd.read_csv(OUT_CSV)
    print(f"[lm] loaded {len(df)} words; columns: {list(df.columns)}")
    for bucket in ["Negative", "Positive", "Uncertainty", "Litigious",
                   "StrongModal", "WeakModal", "Constraining"]:
        if bucket not in df.columns:
            continue
        words = df[df[bucket] > 0]["Word"].astype(str).str.upper().tolist()
        out = REPO_ROOT / "data" / f"lm_{bucket.lower()}_words.txt"
        out.write_text("\n".join(words), encoding="utf-8")
        print(f"[lm] wrote {out} ({len(words)} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
