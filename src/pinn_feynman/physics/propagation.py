"""Propagation-failure statistics of the physics residual (Daw et al., arXiv:2207.02338, sec. 3, App. H).

Heavy tails of the distribution of the residual norms |R(x)| on a uniform grid (high skewness /
kurtosis) signal regions where the solution has not propagated from the boundary data. Used during
training (Trainer, columns of the history).
"""

import numpy as np
from scipy import stats


def propagation_stats(norms):
    """Mean, max, skewness (adjusted Fisher-Pearson) and kurtosis (Fisher) of the residual norms."""
    norms = np.asarray(norms, dtype=float)
    return {"mean": float(np.mean(norms)), "max": float(np.max(norms)),
            "skewness": float(stats.skew(norms, bias=False)), "kurtosis": float(stats.kurtosis(norms, fisher=True))}
