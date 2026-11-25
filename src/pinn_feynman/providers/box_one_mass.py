"""Known master integrals of the one-mass box: the three bubbles, in closed form (s0 = 1, points (s1, s2)).

Same formulas as the bubble lines of `reference.box_one_mass._series_block` (columns 1, 2, 3 = bubble in t, in m^2,
in s), without the polylogarithms of the box. Checked against the reference values.
"""

import numpy as np

from ..reference.box_one_mass import _series_constants, _series_exp, _series_mul
from .base import ClosedFormProvider

ORDERS = (-2, -1, 0, 1, 2)


def _bubbles(points):
    _, bubble = _series_constants()
    s1, s2 = np.asarray(points, dtype=float).reshape(-1, 2).T
    n = len(s1)
    bub = np.broadcast_to(bubble, (n, 4))
    out = np.zeros((n, 3, len(ORDERS)), dtype=complex)
    log_s2 = np.log(s2)
    out[:, 0, 1:] = s2[:, None] * _series_mul(_series_exp(log_s2 - np.log(-s1), 4), bub)
    out[:, 1, 1:] = _series_mul(_series_exp(np.full(n, 1j * np.pi), 4), bub)
    out[:, 2, 1:] = (s1 / s2)[:, None] * _series_mul(_series_exp(log_s2 + 1j * np.pi, 4), bub)
    return out


def bubbles():
    """Provider of the bubbles (columns 1, 2, 3)."""
    return ClosedFormProvider(columns=(1, 2, 3), orders=ORDERS, leading_order=-1, function=_bubbles)
