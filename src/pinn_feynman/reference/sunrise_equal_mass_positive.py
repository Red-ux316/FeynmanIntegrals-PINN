"""Reference values of the equal-mass sunrise master integrals (m^2 = 3) for s > 0, between the singular points of
Boni's (A.10) (s = 0, 3, 27), by integration of the equation along the real axis from one AMFlow point in each
interval.

Same basis, conventions and pole parts as `reference.sunrise_equal_mass` (s < 0):
M = (I_11000, I_11100, I_21100), literal propagators q^2 - m^2, no exp(eps gamma_E), orders eps^-2 .. eps^0.

How the values are obtained:
* F = m^2 (sum x) U - s x1 x2 x3 is positive on the simplex for every real s < 27 = (3m)^2, so the masters are real
  below 27; above 27 they are the limit s + i0;
* in each interval of `S_RANGES` one starting point (`STARTS`): closed-form pole parts and tadpole, finite parts from
  AMFlow + Kira (prescription +i0 on the propagators, i.e. s + i0), read from `starting_values.csv`;
* from there the 6 unknowns are integrated along (A.10) on the real axis with DOP853 (rtol 1e-13) to both ends of
  the interval, read from the dense output; the singular points 0, 3, 27 are never crossed;
* below 27 the imaginary part is set to zero; at eps^-2, eps^-1 (pole parts, polynomials in s) it is zero everywhere.

Limits, said plainly:
* the precision is that of the AMFlow starting data propagated by the equation; AMFlow at other points (s = 10,
  27.1, 80) and a Feynman-parameter integral of the imaginary part above 27 are the independent checks;
* the equation integrated here is the same (A.10) the network solves;
* only s inside one of `S_RANGES`; anything else is an error.
"""

from functools import lru_cache

import numpy as np

from . import sunrise_equal_mass as negative

ORDERS = negative.ORDERS
N_MASTERS = negative.N_MASTERS
MASS_SQUARED = negative.MASS_SQUARED
#: the intervals of s covered, one per gap between the singular points 0, 3, 27 (0.001 away from each of them; the training domains use margins 0.1 and 0.01)
S_RANGES = ((0.001, 2.999), (3.001, 26.999), (27.001, 100.0))
#: one starting point per interval of S_RANGES and the AMFlow finite parts there, (I_11100, I_21100) at eps^0
#: (`starting_values.csv`)
STARTS = tuple((s, finite) for s, finite, _ in negative.read_starts("sunrise_equal_mass_positive"))
THRESHOLD = 27.0
RTOL, ATOL = 1e-13, 1e-15
#: what identifies the output of `reference_values` (enters the cache key of the datasets)
CACHE_PARAMS = {"method": "de_real_axis_from_amflow", "orders": list(ORDERS),
                "starts": [[s, [repr(v) for v in finite]] for s, finite in STARTS],
                "s_ranges": [list(r) for r in S_RANGES], "integrator": "DOP853", "rtol": RTOL, "atol": ATOL}
#: one global integration per process: one chunk, one process (the values never depend on the split)
COMPUTE_OPTIONS = {"chunk_size": 10**6, "max_workers": 1}


def _rhs(s, y):
    """d/ds of the 6 unknowns (I_11100, I_21100) x (eps^-2, eps^-1, eps^0), complex s: A M + source."""
    a = [np.array(f(s), dtype=complex) for f in negative._matrices()]
    tadpole = negative.tadpole_series()
    m = y.reshape(2, 3)
    out = np.zeros((2, 3), dtype=complex)
    for k in range(3):                               # order eps^(k - 2) of the derivative
        for j in range(min(k, 2) + 1):               # A_j acts on the order k - j
            out[:, k] += a[j][1:, 1:] @ m[:, k - j] + a[j][1:, 0] * tadpole[k - j]
    return out.reshape(-1)


def start_values(index):
    """The 6 unknowns at the starting point of S_RANGES[index]: closed-form pole parts, AMFlow finite parts."""
    s0, finite = STARTS[index]
    poles = negative.pole_parts([s0])[0]             # (2 masters, 2 orders)
    return np.array([poles[0, 0], poles[0, 1], finite[0], poles[1, 0], poles[1, 1], finite[1]], dtype=complex)


@lru_cache(maxsize=None)
def _solutions(index):
    """Dense outputs along the real axis, from the AMFlow point of S_RANGES[index] to its two ends (once per process)."""
    from scipy.integrate import solve_ivp

    lo, hi = S_RANGES[index]
    s0 = STARTS[index][0]
    y0 = start_values(index)
    out = []
    for end in (lo, hi):
        sol = solve_ivp(_rhs, (s0, end), y0, method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"integration of (A.10) from s = {s0} to s = {end} failed: {sol.message}")
        out.append(sol.sol)
    return s0, tuple(out)


def _range_index(s):
    s = np.asarray(s)
    if np.iscomplexobj(s):
        raise TypeError("the sunrise reference takes real s")
    s = s.astype(float).reshape(-1)
    if not np.all(np.isfinite(s)):
        raise ValueError("the points contain NaN or infinity")
    index = np.full(len(s), -1)
    for i, (lo, hi) in enumerate(S_RANGES):
        index[(s >= lo) & (s <= hi)] = i
    if np.any(index < 0):
        raise ValueError(f"the sunrise reference for s > 0 covers s in {S_RANGES}; got e.g. s = {s[index < 0][0]:g}")
    return s, index


def reference_values(s, orders=ORDERS):
    """Laurent coefficients of (I_11000, I_11100, I_21100) at the points s: complex array (len(s), 3, len(orders));
    real below 27, s + i0 above."""
    if tuple(orders) != ORDERS:
        raise ValueError(f"the sunrise reference gives the orders {ORDERS} only, got {tuple(orders)}")
    s, index = _range_index(s)
    out = np.zeros((len(s), N_MASTERS, len(ORDERS)), dtype=complex)
    out[:, 0, :] = negative.tadpole_series()
    for i in np.unique(index):
        s0, (towards_lo, towards_hi) = _solutions(int(i))
        for mask, solution in (((index == i) & (s <= s0), towards_lo), ((index == i) & (s > s0), towards_hi)):
            if np.any(mask):
                out[mask, 1:, :] = solution(s[mask]).T.reshape(-1, 2, 3)
    below = s < THRESHOLD
    out[below] = out[below].real
    out[:, :, :2] = out[:, :, :2].real               # pole parts: polynomials in s, real everywhere
    return out
