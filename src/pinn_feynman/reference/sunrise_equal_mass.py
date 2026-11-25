"""Reference values of the equal-mass sunrise master integrals (m^2 = 3, s < 0), by integration of Boni's (A.10).

Basis and conventions of `systems.sunrise_equal_mass`: M = (I_11000, I_11100, I_21100), literal propagators
q^2 - m^2, measure d^dk/(i pi^{d/2}) per loop, no exp(eps gamma_E). For s < 0 and m^2 > 0 the Symanzik polynomial
F is positive: the masters are real, their imaginary part vanishes identically.

How the values are obtained, orders eps^-2 .. eps^0:
* I_11000 = (m^2)^(2 - 2 eps) Gamma(eps - 1)^2, expanded with mpmath (closed form);
* pole parts of I_11100 and I_21100 in closed form (checked against pySecDec at four points):
      I_11100 = 9/(2 eps^2) - [9 (gamma - 3/2 + ln 3) + s/4]/eps + ...,
      I_21100 = 1/(2 eps^2) - [gamma - 1/2 + ln 3]/eps + ...;
* the finite parts at the starting point s = -1 are pySecDec's (epsrel 1e-9 requested, declared error 4e-13 and
  9e-14), read from `starting_values.csv` (`read_starts`);
* the 6 unknowns (I_11100, I_21100 at three orders) are then integrated along (A.10) in t = ln(-s) from s = -1 to
  both ends of `S_RANGE` with DOP853 (rtol 1e-13), once per process, and read from the dense output.

Limits, said plainly:
* the precision is that of the starting data (pySecDec, declared ~1e-13 absolute on the finite parts) propagated by
  the equation; against three other pySecDec points (s = -0.01, -10, -100) the mixed error is <= 5.5e-12;
* the equation integrated here is the same (A.10) the network solves: these values are not an independent check of
  the matrix;
* only s in S_RANGE = [-100, -0.01], the range checked against pySecDec; anything else is an error.

`tadpole_series`, `pole_parts` and `read_starts` are also used by `sunrise_equal_mass_positive` (s > 0).
"""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Laurent orders of Boni's sunrise: eps^-2 ... eps^0
ORDERS = (-2, -1, 0)
N_MASTERS = 3
MASS_SQUARED = 3.0
#: the file of the starting values (finite parts at eps^0, with their error or digits and their source)
STARTING_VALUES = Path(__file__).with_name("starting_values.csv")
#: the two masters integrated from a starting point, in the order of the basis
INTEGRATED_MASTERS = ("I_11100", "I_21100")


def read_starts(problem):
    """((s, (value_11100, value_21100), (error_11100, error_21100)), ...) at eps^0 for `problem`, from
    `starting_values.csv`, sorted by s. A value with zero imaginary part is a float; an empty error is None."""
    points = {}
    with open(STARTING_VALUES, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["problem"] != problem:
                continue
            if row["order"] != "0":
                raise ValueError(f"{STARTING_VALUES.name}: only finite parts (order 0), got order {row['order']}")
            real, imag = float(row["real"]), float(row["imag"])
            value = real if imag == 0 else complex(real, imag)
            error = float(row["error"]) if row["error"] else None
            points.setdefault(float(row["s"]), {})[row["master"]] = (value, error)
    if not points:
        raise ValueError(f"{STARTING_VALUES.name} has no starting value for {problem!r}")
    out = []
    for s in sorted(points):
        if set(points[s]) != set(INTEGRATED_MASTERS):
            raise ValueError(f"{STARTING_VALUES.name}: {problem} at s = {s:g} needs {INTEGRATED_MASTERS}, "
                             f"got {sorted(points[s])}")
        out.append((s, tuple(points[s][m][0] for m in INTEGRATED_MASTERS),
                    tuple(points[s][m][1] for m in INTEGRATED_MASTERS)))
    return tuple(out)


#: starting point of the integration and the finite parts of pySecDec there, with their declared errors
((START, START_FINITE, START_FINITE_SIGMA),) = read_starts("sunrise_equal_mass")
#: the range of s checked against pySecDec; points outside it are refused
S_RANGE = (-100.0, -0.01)
RTOL, ATOL = 1e-13, 1e-15
#: what identifies the output of `reference_values` (enters the cache key of the datasets: do not change the content)
CACHE_PARAMS = {"method": "de_integration", "orders": list(ORDERS), "start": START,
                "start_finite": [repr(v) for v in START_FINITE], "s_range": list(S_RANGE),
                "integrator": "DOP853", "rtol": RTOL, "atol": ATOL}
#: one global integration per process: one chunk, one process (the values never depend on the split)
COMPUTE_OPTIONS = {"chunk_size": 10**6, "max_workers": 1}


@lru_cache(maxsize=None)
def tadpole_series():
    """(I_11000^(-2), I_11000^(-1), I_11000^(0)): 3^(2 - 2 eps) Gamma(eps - 1)^2 = 3^(2 - 2 eps) Gamma(1 + eps)^2 /
    (eps^2 (eps - 1)^2), expanded at 30 digits."""
    import mpmath as mp

    with mp.workdps(30):
        m2 = mp.mpf(MASS_SQUARED)
        c = mp.taylor(lambda e: m2 ** (2 - 2 * e) * mp.gamma(1 + e) ** 2 / (e - 1) ** 2, 0, len(ORDERS) - 1)
        return tuple(float(v) for v in c)


def pole_parts(s):
    """Pole parts of I_11100 and I_21100 at the points s: array (N, 2, 2), orders eps^-2, eps^-1."""
    s = np.asarray(s, dtype=float).reshape(-1)
    g, log_m2 = float(np.euler_gamma), float(np.log(MASS_SQUARED))
    out = np.empty((len(s), 2, 2))
    out[:, 0, 0] = 3 * MASS_SQUARED / 2
    out[:, 0, 1] = -(3 * MASS_SQUARED * (g - 1.5 + log_m2) + s / 4)
    out[:, 1, 0] = 0.5
    out[:, 1, 1] = -(g - 0.5 + log_m2)
    return out


@lru_cache(maxsize=None)
def _matrices():
    from ..systems import sunrise_equal_mass as system

    return system.numeric_orders("numpy")


def _rhs(t, y):
    """d/dt of the 6 unknowns (I_11100, I_21100) x (eps^-2, eps^-1, eps^0), t = ln(-s): s (A M + source)."""
    s = -np.exp(t)
    a = [np.array(f(s), dtype=float) for f in _matrices()]
    tadpole = tadpole_series()
    m = y.reshape(2, 3)
    out = np.zeros((2, 3))
    for k in range(3):                               # order eps^(k - 2) of the derivative
        for j in range(min(k, 2) + 1):               # A_j acts on the order k - j
            out[:, k] += a[j][1:, 1:] @ m[:, k - j] + a[j][1:, 0] * tadpole[k - j]
    return s * out.reshape(-1)


def _start_values():
    poles = pole_parts([START])[0]                   # (2 masters, 2 orders)
    return np.array([poles[0, 0], poles[0, 1], START_FINITE[0], poles[1, 0], poles[1, 1], START_FINITE[1]])


@lru_cache(maxsize=None)
def _solutions():
    """Dense outputs from s = START to the two ends of S_RANGE (computed once per process)."""
    from scipy.integrate import solve_ivp

    t0, y0 = np.log(-START), _start_values()
    out = []
    for end in S_RANGE:
        sol = solve_ivp(_rhs, (t0, np.log(-end)), y0, method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"integration of (A.10) towards s = {end} failed: {sol.message}")
        out.append(sol.sol)
    return tuple(out)


def _check_points(s):
    s = np.asarray(s)
    if np.iscomplexobj(s):
        raise TypeError("the sunrise reference takes real s < 0")
    s = s.astype(float).reshape(-1)
    if not np.all(np.isfinite(s)):
        raise ValueError("the points contain NaN or infinity")
    lo, hi = S_RANGE
    bad = (s < lo) | (s > hi)
    if np.any(bad):
        raise ValueError(f"the sunrise reference covers s in [{lo:g}, {hi:g}] (checked against pySecDec); got e.g. "
                         f"s = {s[bad][0]:g}")
    return s


def reference_values(s, orders=ORDERS):
    """Laurent coefficients of (I_11000, I_11100, I_21100) at the points s: complex array (len(s), 3, len(orders)),
    with zero imaginary part (s < 0, m^2 > 0)."""
    if tuple(orders) != ORDERS:
        raise ValueError(f"the sunrise reference gives the orders {ORDERS} only, got {tuple(orders)}")
    s = _check_points(s)
    out = np.zeros((len(s), N_MASTERS, len(ORDERS)), dtype=complex)
    out[:, 0, :] = tadpole_series()
    t = np.log(-s)
    towards_far, towards_zero = _solutions()         # S_RANGE[0] is the far end (|s| > 1), S_RANGE[1] near s = 0
    far = s <= START
    for mask, solution in ((far, towards_far), (~far, towards_zero)):
        if np.any(mask):
            out[mask, 1:, :] = solution(t[mask]).T.reshape(-1, 2, 3)
    return out
