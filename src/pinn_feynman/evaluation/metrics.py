"""Error metrics of Boni, thesis eqs. (6.18)-(6.19).

Arrays have shape (N_test, n_masters, n_orders); the statistics are taken over the points
(and over master integrals or orders, as in eq. 6.19). Relative errors exclude the
points where the true value vanishes (thesis p. 98).
"""

import numpy as np

#: |y_test| below this is treated as zero in relative errors
ZERO_THRESHOLD = 1e-10


def _same_shape(prediction, target):
    """Refuse shapes that numpy would silently broadcast (a network with fewer masters than the dataset)."""
    if np.shape(prediction) != np.shape(target):
        raise ValueError(f"prediction of shape {np.shape(prediction)} and target of shape {np.shape(target)}: "
                         "compare on the columns of the run (`Run.target`)")


def absolute_error(prediction, target):
    """Eq. (6.18): |y_nn - y_test|, element by element."""
    _same_shape(prediction, target)
    return np.abs(prediction - target)


def relative_error(prediction, target, threshold=ZERO_THRESHOLD):
    """Eq. (6.18): |y_nn - y_test| / |y_test|, NaN where |y_test| < threshold."""
    target = np.asarray(target)
    _same_shape(prediction, target)
    out = np.full(target.shape, np.nan)
    ok = np.abs(target) >= threshold
    out[ok] = np.abs(prediction[ok] - target[ok]) / np.abs(target[ok])
    return out


def _stats(values):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, "count": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max()), "count": int(v.size)}


def summarize(errors, orders):
    """Statistics of an error array (N, n_masters, n_orders).

    by_order: over master integrals and points (eq. 6.19, first line);
    by_master: over orders and points (eq. 6.19, second line);
    by_component: over points, for each (master, order); overall: everything.
    """
    n_masters = errors.shape[1]
    return {
        "overall": _stats(errors.ravel()),
        "by_order": {str(o): _stats(errors[:, :, k].ravel()) for k, o in enumerate(orders)},
        "by_master": {f"M{i + 1}": _stats(errors[:, i, :].ravel()) for i in range(n_masters)},
        "by_component": {f"M{i + 1}^({o})": _stats(errors[:, i, k])
                         for i in range(n_masters) for k, o in enumerate(orders)},
    }
