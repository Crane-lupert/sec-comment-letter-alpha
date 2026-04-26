"""Universe management: Russell 3000 CIK list construction.

The universe parquet is the single source of truth for which CIKs we enqueue.
Day 1: bootstrap-from-SEC (one-shot static metadata, see scripts/bootstrap_universe.py).
Day 2+: refine to actual Russell 3000 via IWV holdings or CRSP/Compustat link.

Schema of `data/universe_ciks.parquet`:
    cik:       str (10-char zero-padded)
    ticker:    str
    name:      str
    source:    str (e.g. "sec_company_tickers")
    bootstrap_ts: str (ISO 8601 UTC)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PARQUET_LEGACY = REPO_ROOT / "data" / "universe_ciks.parquet"  # SEC-tickers, sorted by CIK
UNIVERSE_PARQUET_R3K = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"  # actual Russell 3000 (IWV)
UNIVERSE_PARQUET = UNIVERSE_PARQUET_R3K  # default for new code paths
DEFAULT_UNIVERSE = UNIVERSE_PARQUET  # alias


def load_universe(path: Path | None = None) -> pd.DataFrame:
    """Load the canonical universe parquet (R3K by default, legacy as fallback).

    Day 3 audit found the legacy SEC-ticker-sorted universe was biased to old
    firms. R3K is now the source of truth; legacy is preserved only because
    the old enqueue+features predate the rebuild.
    """
    p = Path(path) if path else UNIVERSE_PARQUET
    if not p.exists() and path is None and UNIVERSE_PARQUET_LEGACY.exists():
        # auto-fallback so old test paths still work
        p = UNIVERSE_PARQUET_LEGACY
    if not p.exists():
        raise FileNotFoundError(
            f"Universe parquet missing: {p}\n"
            f"Run: uv run python scripts/bootstrap_universe_r3k.py"
        )
    return pd.read_parquet(p)


def sample_ciks(df: pd.DataFrame, n: int, *, seed: int = 17) -> list[str]:
    if len(df) <= n:
        return df["cik"].astype(str).tolist()
    return df.sample(n=n, random_state=seed)["cik"].astype(str).tolist()
