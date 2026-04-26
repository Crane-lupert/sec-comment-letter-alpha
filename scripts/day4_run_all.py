"""Day 4 orchestrator: build_pairs -> build_panel -> construct_signal -> orthogonalize.

Run after all background extractions + yfinance fetch + daemon refetch are done.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

STEPS = [
    ("rebuild day4 pairs (UPLOAD+CORRESP join)", "scripts/day4_build_pairs.py"),
    ("build firm-month panel + per-event BHAR/CAR", "scripts/day4_build_panel.py"),
    ("construct Signal A/B monthly factor returns", "scripts/day4_construct_signal.py"),
    ("orthogonalize against FF5+UMD + IS/OOS split", "scripts/day4_orthogonalize.py"),
]


def main() -> int:
    for label, script in STEPS:
        print(f"\n{'='*70}")
        print(f"# {label}")
        print(f"# python {script}")
        print(f"{'='*70}")
        r = subprocess.run([str(PYTHON), script], cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"\n[day4-orchestrator] FATAL: {script} returned {r.returncode}, aborting.")
            return r.returncode
    print("\n[day4-orchestrator] all steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
