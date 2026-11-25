"""Known master integral of the equal-mass sunrise: the product of two tadpoles I_11000, constant in s.

I_11000 = (m^2)^(2 - 2 eps) Gamma(eps - 1)^2 with m^2 = 3 (literal convention of `systems.sunrise_equal_mass`), real,
series from eps^-2: the same numbers as `reference.sunrise_equal_mass.tadpole_series`.
"""

import numpy as np

from .base import ClosedFormProvider

ORDERS = (-2, -1, 0)


def _tadpole(v):
    from ..reference.sunrise_equal_mass import tadpole_series       # mpmath only on the first call

    value = np.asarray(tadpole_series(), dtype=complex)
    return np.broadcast_to(value, (len(v), 1, len(ORDERS))).copy()


def tadpole():
    """Provider of the tadpole I_11000 (column 0), constant in s, leading order eps^-2."""
    return ClosedFormProvider(columns=(0,), orders=ORDERS, leading_order=-2, function=_tadpole)
