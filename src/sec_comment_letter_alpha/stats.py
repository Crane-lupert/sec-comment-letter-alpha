"""Cross-section statistical rigor: FF5+momentum+PEAD+LM orthogonalization,
FDR (Benjamini-Hochberg), Deflated Sharpe Ratio (Bailey-Lopez de Prado),
month-clustered bootstrap.

Day 1: API skeletons. Implementation lands Day 4-7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorReturn:
    dates: pd.DatetimeIndex
    returns: pd.Series  # monthly
    name: str


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. Returns rejection mask (bool)."""
    raise NotImplementedError("Day 6-7 rigor pass")


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> float:
    """Bailey-Lopez de Prado (2014) DSR. n_trials = total candidate signals tested."""
    raise NotImplementedError("Day 6-7 rigor pass")


def newey_west_se(residuals: np.ndarray, lags: int = 6) -> float:
    raise NotImplementedError("Day 5")


def cluster_bootstrap_ci(
    returns: pd.Series,
    *,
    cluster: pd.Series,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    raise NotImplementedError("Day 6-7 rigor pass")


def orthogonalize(
    target: pd.Series,
    factors: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """OLS regress target on factors. Returns (residual_series, betas)."""
    raise NotImplementedError("Day 4-5")
