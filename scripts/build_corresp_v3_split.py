"""Build the v3-corresp held-out split (train/test) from existing R3K CORRESP records.

Deterministic: seed=42 random sample. Run ONCE, output is committed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "day3_corresp_features_r3k.jsonl"
OUT = REPO_ROOT / "docs" / "preregistration" / "v3_corresp_split.json"
TRAIN_N = 100
SEED = 42


def main() -> int:
    keys: list[list[str]] = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "skip" in obj or "fatal_error" in obj:
            continue
        keys.append([obj["cik"], obj["accession"]])
    print(f"[split] source records: {len(keys)}")

    rng = random.Random(SEED)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    train = shuffled[:TRAIN_N]
    test = shuffled[TRAIN_N:]
    print(f"[split] train={len(train)}  test={len(test)}  seed={SEED}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "train": train, "test": test}, indent=2),
                   encoding="utf-8")
    print(f"[split] wrote {OUT}")

    train_path = OUT.parent / "v3_corresp_split_train.json"
    test_path = OUT.parent / "v3_corresp_split_test.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    test_path.write_text(json.dumps(test), encoding="utf-8")
    print(f"[split] wrote {train_path}")
    print(f"[split] wrote {test_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
