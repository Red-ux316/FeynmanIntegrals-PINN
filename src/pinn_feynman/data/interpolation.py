"""Chebyshev interpolation on an interval (for the large test set).

The Laurent coefficients of the massless box are analytic on the training domain
(singular points x = 0, -1 lie outside it), so the Chebyshev interpolant converges
geometrically; the accuracy is checked against exactly computed points.
"""

import numpy as np


def chebyshev_nodes(domain, n):
    """n Chebyshev points of the second kind on [lo, hi], end points included, increasing."""
    lo, hi = domain
    k = np.arange(n)
    return (lo + hi) / 2 - (hi - lo) / 2 * np.cos(np.pi * k / (n - 1))


def _barycentric_weights(n):
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def barycentric_interpolate(nodes, values, xs):
    """Interpolant through (nodes, values) evaluated at xs (barycentric formula, 2nd kind).

    `nodes` must be the output of chebyshev_nodes; `values` has shape (n, ...) and may be
    complex; the result has shape (len(xs), ...).
    """
    xs = np.asarray(xs, dtype=float)
    w = _barycentric_weights(len(nodes))
    diff = xs[:, None] - nodes[None, :]
    exact = diff == 0
    diff[exact] = 1.0
    c = w[None, :] / diff
    tail = values.shape[1:]
    flat = values.reshape(len(nodes), -1)
    out = (c @ flat) / c.sum(axis=1, keepdims=True)
    rows, cols = np.nonzero(exact)
    out[rows] = flat[cols]
    return out.reshape(len(xs), *tail)
