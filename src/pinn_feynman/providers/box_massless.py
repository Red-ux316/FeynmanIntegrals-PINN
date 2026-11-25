"""Known master integrals of the massless box: the two bubbles, in closed form (s = 1, x = t/s < 0).

Basis of `systems.box_massless` (Boni eq. 6.26): M1 = e^{i pi eps} B(eps) (constant in x) and
M2 = |x|^eps e^{2 i pi eps} B(eps), with B(eps) = e^{eps gamma_E} Gamma(eps) Gamma(1-eps)^2 / Gamma(2-2 eps), whose
series starts at eps^-1. Checked against `reference.box_massless.laurent_coefficients`.
"""

import numpy as np

from ..reference.box_one_mass import _series_constants, _series_exp, _series_mul
from .base import ClosedFormProvider

ORDERS = (-2, -1, 0, 1, 2)


def _bubbles(x):
    _, bubble = _series_constants()
    x = np.asarray(x, dtype=float).reshape(-1)
    bub = np.broadcast_to(bubble, (len(x), 4))
    out = np.zeros((len(x), 2, len(ORDERS)), dtype=complex)
    out[:, 0, 1:] = _series_mul(_series_exp(np.full(len(x), 1j * np.pi), 4), bub)
    out[:, 1, 1:] = _series_mul(_series_exp(np.log(np.abs(x)) + 2j * np.pi, 4), bub)
    return out


def bubbles():
    """Provider of the bubbles M1, M2 (columns 0 and 1)."""
    return ClosedFormProvider(columns=(0, 1), orders=ORDERS, leading_order=-1, function=_bubbles)
