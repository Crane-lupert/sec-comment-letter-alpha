"""Day 7 FDR helper unit tests — Benjamini-Hochberg correctness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "day7_fdr", REPO_ROOT / "scripts" / "day7_fdr.py"
)
day7 = importlib.util.module_from_spec(SPEC)
sys.modules["day7_fdr"] = day7
SPEC.loader.exec_module(day7)  # type: ignore[attr-defined]


def test_bh_classic_worked_example():
    # Classic 10-test BH example. With α=0.05, rejections should be the smallest
    # 2 hypotheses (p=0.001 -> q=0.01, p=0.008 -> q=0.04).
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    bh, passes = day7.benjamini_hochberg(ps, alpha=0.05)
    assert passes[0] is True
    assert passes[1] is True
    assert all(p is False for p in passes[2:])
    # BH adjusted p must be monotone non-decreasing in p-rank
    paired = sorted(zip(ps, bh), key=lambda kv: kv[0])
    for (_, q1), (_, q2) in zip(paired, paired[1:]):
        assert q2 + 1e-12 >= q1


def test_bh_all_significant():
    ps = [0.001, 0.002, 0.003, 0.004]
    bh, passes = day7.benjamini_hochberg(ps, alpha=0.05)
    assert all(passes)
    # The largest BH-adjusted p is p_max (since q_n = p_n * n/n = p_n)
    assert abs(bh[3] - 0.004) < 1e-12


def test_bh_handles_nans():
    # Two NaNs and three real p-values; BH ignores NaN entries.
    ps = [float("nan"), 0.01, 0.5, float("nan"), 0.04]
    bh, passes = day7.benjamini_hochberg(ps, alpha=0.05)
    # NaNs map to NaN BH and False pass
    assert np.isnan(bh[0]) and bh[3] != bh[3]  # NaN check
    assert passes[0] is False and passes[3] is False
    # Smallest finite p=0.01, rank 1 of 3 -> q=0.03 -> pass
    assert passes[1] is True
    assert abs(bh[1] - 0.03) < 1e-12


def test_bh_empty_input_safe():
    bh, passes = day7.benjamini_hochberg([], alpha=0.05)
    assert bh == [] and passes == []
