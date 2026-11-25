"""Reference values of the non-planar three-point masters (m10, m11) (Boni sec. 6.3.3), x = -s/m^2, s -> s + i0.

Basis and system of `systems.nonplanar_three_point` (von Manteuffel-Tancredi arXiv:1701.05905, "vMT"). The masters
start at eps^4 (vMT sec. 4.1): the orders eps^0 .. eps^3 are exactly zero, and at eps^4
    dm/dx = B(x) m + (0, N11^(4)(x)),
with the forcing N11^(4) of vMT eqs. (4.33)-(4.36) on the real axis (`forcing`), x -> x - i0.

How the values are obtained:
* at three starting points x = 1, -2, -10, by variation of constants with the homogeneous solutions of vMT
  (4.18), (4.22) continued to x - i0 and the integration constants of vMT (4.40) (no sqrt(x) branch at x = 0):
  `variation_of_constants`, mpmath at 20 digits, about 2 s for the three points, once per process. At x = -5 this
  reproduces the 50-digit value of vMT (4.44) to 1.7e-31 (at 30 digits);
* from each starting point the equation is integrated with DOP853 (rtol 1e-13) in a variable in which the forcing
  is smooth: t = ln|x| towards x = 0, u = sqrt|x + 4| towards the threshold x = -4 (the forcing has sqrt(x + 4)
  branches there), and the dense outputs are read at the points. The two sides of x = -4 come from different
  starting points: their agreement at x = -4 is a check.

Limits, said plainly:
* x in X_RANGE = [-30, 1], x = 0 included (value 0, the limit); 0 < |x| < X_SMALLEST is refused;
* the equation integrated here is the one the network solves (same B, same forcing): the values are not an
  independent check of them. The independent checks were: the forcing against the continuation of vMT (4.2),
  and the solution against vMT (4.44) and Fig. 1.
"""

from functools import lru_cache

import numpy as np

ORDERS = (0, 1, 2, 3, 4)
N_MASTERS = 2
#: the only non-zero order (vMT sec. 4.1)
LEADING_ORDER = 4
X_RANGE = (-30.0, 1.0)
X_SMALLEST = 1e-12
STARTS = (1.0, -2.0, -10.0)
THRESHOLD = -4.0
RTOL, ATOL = 1e-13, 1e-16
START_DIGITS = 20
#: what identifies the output of `reference_values` (enters the cache key of the datasets: do not change the content)
CACHE_PARAMS = {"method": "vmt_variation_of_constants+de_integration", "orders": list(ORDERS),
                "starts": list(STARTS), "start_digits": START_DIGITS, "x_range": list(X_RANGE),
                "x_smallest": X_SMALLEST, "integrator": "DOP853", "rtol": RTOL, "atol": ATOL}
#: one global integration per process: one chunk, one process (the values never depend on the split)
COMPUTE_OPTIONS = {"chunk_size": 10**6, "max_workers": 1}

ZETA2 = np.pi**2 / 6


# --- forcing term ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _clausen_coefficients(terms=30):
    """|B_2k| / (2k (2k + 1)!), k = 1..terms (series of Cl2 at 0)."""
    from math import factorial

    from scipy.special import bernoulli

    b = bernoulli(2 * terms)
    return np.array([abs(b[2 * k]) / (2 * k * factorial(2 * k + 1)) for k in range(1, terms + 1)])


def clausen2(theta):
    """Cl2(theta) for |theta| <= pi: theta - theta ln|theta| + sum_k |B_2k| theta^(2k+1) / (2k (2k+1)!)."""
    theta = np.asarray(theta, dtype=float)
    c = _clausen_coefficients()
    t2 = theta * theta
    series = np.zeros_like(theta)
    for coefficient in c[::-1]:
        series = (series + coefficient) * t2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_part = np.where(theta == 0, 0.0, theta * np.log(np.abs(theta)))
    return theta - log_part + theta * series


def _li2_of_exp(a):
    """Li2(exp(-a)) for a >= 0, through scipy's spence(z) = Li2(1 - z) with 1 - exp(-a) computed by expm1."""
    from scipy.special import spence

    return spence(-np.expm1(-a))


def forcing(x):
    """N11^(4)(x - i0) on the real axis, x < 4: complex array (vMT 4.34 for 0 < x < 4, 4.35 for -4 <= x < 0, 4.36 for
    x < -4; 0 at x = 0). The logarithms of the letters are written as asinh, the arcsec as atan2, to keep the precision
    near 0 and -4."""
    x = np.asarray(x, dtype=float)
    if np.any(x >= 4) or not np.all(np.isfinite(x)):
        raise ValueError("the forcing term is written for finite x < 4")
    out = np.zeros(x.shape, dtype=complex)
    pos, mid, low = x > 0, (x < 0) & (x >= THRESHOLD), x < THRESHOLD

    xp = x[pos]                                                      # 0 < x < 4
    ln_l2 = np.arcsinh(np.sqrt(xp) / 2)
    out[pos] = 5 * ln_l2**2 + 3 * np.sqrt(xp) * clausen2(-2 * np.arcsin(np.sqrt(xp) / 2)) / np.sqrt(4 - xp)

    for mask in (mid, low):
        xn = x[mask]
        p1, p5 = np.sqrt(-xn), np.sqrt(4 - xn)
        ln_p4 = np.arcsinh(p1 / 2)                                   # l'_4 = (sqrt(-x) + sqrt(4 - x)) / 2
        common = 3 * p1 * (ZETA2 - ln_p4**2 - _li2_of_exp(2 * ln_p4)) / p5
        imag = 3 * np.pi * p1 / p5 * ln_p4
        if mask is mid:                                              # arcsec(2 / l3) = arctan2(sqrt(-x), sqrt(x + 4))
            real = -5 * np.arctan2(p1, np.sqrt(xn + 4)) ** 2 + common
        else:
            ln_p2 = np.arcsinh(np.sqrt(-xn - 4) / 2)                 # l'_2 = (sqrt(-x) + sqrt(-x - 4)) / 2
            real = 5 * ln_p2**2 - 7.5 * ZETA2 + common
            imag = imag - 5 * np.pi * ln_p2
        out[mask] = real + 1j * imag
    return out


# --- starting values (mpmath) ----------------------------------------------------------------------------------------
def _forcing_mp(y, mp):
    """N11^(4)(y - i0) in mpmath, the same region formulas as `forcing` (vMT 4.34-4.36)."""
    z2 = mp.zeta(2)
    if y > 0:
        return 5 * mp.asinh(mp.sqrt(y) / 2) ** 2 + 3 * mp.sqrt(y) * mp.clsin(2, -2 * mp.asin(mp.sqrt(y) / 2)) / mp.sqrt(4 - y)
    p1, p5 = mp.sqrt(-y), mp.sqrt(4 - y)
    ln_p4 = mp.asinh(p1 / 2)
    common = 3 * p1 * (z2 - ln_p4**2 - mp.polylog(2, mp.exp(-2 * ln_p4))) / p5
    imag = 3 * mp.pi * p1 / p5 * ln_p4
    if y >= THRESHOLD:
        return -5 * mp.atan2(p1, mp.sqrt(y + 4)) ** 2 + common + 1j * imag
    ln_p2 = mp.asinh(mp.sqrt(-y - 4) / 2)
    return 5 * ln_p2**2 - mp.mpf(15) / 2 * z2 + common + 1j * (imag - 5 * mp.pi * ln_p2)


def _homogeneous_mp(y, mp):
    """G^(0,16) of vMT (4.18), (4.22) continued to y - i0: sqrt(y - i0), K and E at 1 - y/16 + i0 for y < 0."""
    tiny = mp.mpf(10) ** (-2 * mp.mp.dps)
    root = mp.sqrt(y) if y > 0 else -1j * mp.sqrt(-y)
    m, mc = y / 16, 1 - y / 16
    k, e = mp.ellipk(m), mp.ellipe(m)
    kc = mp.ellipk(mc) if mc < 1 else mp.ellipk(mp.mpc(mc, tiny))
    ec = mp.ellipe(mc) if mc < 1 else mp.ellipe(mp.mpc(mc, tiny))
    return ((root * k, root * kc), (-root * e, root * (ec - kc)))


def variation_of_constants(x, digits=START_DIGITS):
    """(m10^(4), m11^(4)) at a real x != 0 (complex for x < 0), vMT (4.40) continued to x - i0 along the real axis:
    m = (2/pi) [J(x) int_0^x I1 N / y - I(x) int_0^x J1 N / y] (columns I, J of G)."""
    import mpmath as mp

    with mp.workdps(digits):
        x = mp.mpf(x)
        nodes = [0, x] if x > THRESHOLD else [0, THRESHOLD, x]
        a = mp.quad(lambda y: _homogeneous_mp(y, mp)[0][0] * _forcing_mp(y, mp) / y, nodes)
        b = mp.quad(lambda y: _homogeneous_mp(y, mp)[0][1] * _forcing_mp(y, mp) / y, nodes)
        g = _homogeneous_mp(x, mp)
        m10 = 2 / mp.pi * (g[0][1] * a - g[0][0] * b)
        m11 = 2 / mp.pi * (g[1][1] * a - g[1][0] * b)
        return complex(m10), complex(m11)


# --- integration ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _b_matrix():
    from ..systems import nonplanar_three_point as system

    return system.numeric_orders("numpy")[0]


def _dmdx(x, y):
    return np.array(_b_matrix()(x), dtype=float) @ y + np.array([0.0, forcing(np.array([x]))[0]])


def _segments():
    """(variable -> x, dx/dvariable, variable at the start, variable at the end, start x) of the five pieces."""
    return {
        "positive": (np.exp, np.exp, 0.0, np.log(X_SMALLEST), 1.0),                           # t = ln x
        "inner": (lambda t: -np.exp(t), lambda t: -np.exp(t), np.log(2.0), np.log(X_SMALLEST), -2.0),  # t = ln(-x)
        "above": (lambda u: -4 + u * u, lambda u: 2 * u, np.sqrt(2.0), 0.0, -2.0),           # u = sqrt(x + 4)
        "below": (lambda u: -4 - u * u, lambda u: -2 * u, np.sqrt(6.0), 0.0, -10.0),         # u = sqrt(-x - 4)
        "outer": (lambda t: -np.exp(t), lambda t: -np.exp(t), np.log(10.0), np.log(-X_RANGE[0]), -10.0),
    }


@lru_cache(maxsize=None)
def start_values():
    """{x: (m10^(4), m11^(4))} at the starting points (mpmath, once per process)."""
    return {x: variation_of_constants(x) for x in STARTS}


@lru_cache(maxsize=None)
def _solutions():
    """Dense outputs of the five pieces (computed once per process)."""
    from scipy.integrate import solve_ivp

    starts = start_values()
    out = {}
    for name, (to_x, dx, v0, v1, x0) in _segments().items():
        def rhs(v, y, to_x=to_x, dx=dx):
            return _dmdx(float(to_x(v)), y) * dx(v)

        y0 = np.array(starts[x0], dtype=complex)
        sol = solve_ivp(rhs, (v0, v1), y0, method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"integration of the piece {name!r} failed: {sol.message}")
        out[name] = sol.sol
    return out


def _check_points(x):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise TypeError("the non-planar reference takes real x (the i0 is fixed: x - i0)")
    x = x.astype(float).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("the points contain NaN or infinity")
    lo, hi = X_RANGE
    bad = (x < lo) | (x > hi) | ((x != 0) & (np.abs(x) < X_SMALLEST))
    if np.any(bad):
        raise ValueError(f"the non-planar reference covers x in [{lo:g}, {hi:g}] with x = 0 or |x| >= {X_SMALLEST:g}; "
                         f"got e.g. x = {x[bad][0]:g}")
    return x


def leading_values(x):
    """(m10^(4), m11^(4)) at the points x: complex array (len(x), 2)."""
    x = _check_points(x)
    out = np.zeros((len(x), N_MASTERS), dtype=complex)
    pieces = _solutions()
    masks = {
        "positive": (x > 0, lambda v: np.log(v)),
        "inner": ((x < 0) & (x > -2), lambda v: np.log(-v)),
        "above": ((x <= -2) & (x >= THRESHOLD), lambda v: np.sqrt(v + 4)),
        "below": ((x < THRESHOLD) & (x >= -10), lambda v: np.sqrt(-v - 4)),
        "outer": (x < -10, lambda v: np.log(-v)),
    }
    for name, (mask, variable) in masks.items():
        if np.any(mask):
            out[mask] = pieces[name](variable(x[mask])).T
    return out


def reference_values(x, orders=ORDERS):
    """Taylor coefficients eps^0 .. eps^4 of (m10, m11) at the points x: complex array (len(x), 2, 5); only eps^4 is
    non-zero."""
    if tuple(orders) != ORDERS:
        raise ValueError(f"the non-planar reference gives the orders {ORDERS} only, got {tuple(orders)}")
    x = _check_points(x)
    out = np.zeros((len(x), N_MASTERS, len(ORDERS)), dtype=complex)
    out[:, :, ORDERS.index(LEADING_ORDER)] = leading_values(x)
    return out
