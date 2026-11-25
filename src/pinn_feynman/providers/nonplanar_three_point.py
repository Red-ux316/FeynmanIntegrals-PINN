"""Forcing term of the non-planar three-point masters (m10, m11): the subtopologies enter only through N = (0, N11).

N11 has a single non-zero Taylor coefficient, N11^(4) (vMT arXiv:1701.05905 eq. 4.2), continued to the real axis with
x -> x - i0 (vMT 4.34-4.36): `reference.nonplanar_three_point.forcing`, numpy, about 0.3 microseconds per point.
"""

import numpy as np

from .base import ClosedFormForcing

ORDERS = (0, 1, 2, 3, 4)


def _forcing(v):
    from ..reference.nonplanar_three_point import forcing

    out = np.zeros((len(v), 2, len(ORDERS)), dtype=complex)
    out[:, 1, ORDERS.index(4)] = forcing(v)
    return out


def source():
    """Provider of the forcing term of (m10, m11), rows 0 and 1 of the basis, orders eps^0 .. eps^4."""
    return ClosedFormForcing(rows=(0, 1), orders=ORDERS, function=_forcing)
