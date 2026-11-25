"""Exact reference values of the one-loop one-mass box master integrals (physical region).

Ground truth for training (boundary points) and testing of the one-mass box of Boni's
thesis (eqs. 6.29-6.38, matrices A.4-A.5). The checks of the formulas are listed in the README of
this folder.

Kinematics (Boni A.3 and region 6.38): s0 = s > 0, s1 = t < 0, s2 = m^2 > 0 and
s0 + s1 - s2 > 0 (u = s2 - s0 - s1 < 0), the physical s-channel region (Boni 6.38). With mu = 1
the basis is scale invariant, so s0 = 1 without loss of generality: `reference_values`
takes points (s1, s2) of the open triangle s1 < 0, s2 > 0, 1 + s1 - s2 > 0.

Basis, in the order of Boni's matrices A.4-A.5:
    M = e^{eps gamma_E} ( s1 s2^{1+eps} G_1111,   s2^{1+eps} / s0 G_bub(s1),
                          s2^eps G_bub(s2),        s1 s2^{-1+eps} G_bub(s0) ),
    G_bub(X) = (-X - i0)^{-eps} Gamma(eps) Gamma(1-eps)^2 / Gamma(2-2eps)   (Henn eq. 3.17),
with G_1111 the one-mass box of BDK (hep-ph/9306240) eq. (4.26); measure d^Dk/(i pi^{D/2}).

Box: BDK (4.36), rewritten and continued to the region with the Feynman prescription,
    G_1111 = 2 r_G / (eps^2 s t) b^eps [ (1-x)^eps F(z1)
                                         + e^{i pi eps} (y-1)^eps (F(z2) - (1-x)^eps F(z3)) ],
    x = t/m^2, y = s/m^2, b = -m^2/(s t), F(z) = 2F1(-eps, -eps; 1-eps; z),
    z1 = 1 - y/(1-x), z2 = 1 - x/(1-y), z3 = (1-x-y)/((1-x)(1-y)),
    r_G = Gamma(1+eps) Gamma(1-eps)^2 / Gamma(1-2eps),
where b, 1-x, y-1 > 0 and z1, z2, z3 < 1, so that every 2F1 is real. The side of the cut is
the explicit phase e^{i pi eps}; it is never obtained from the log of a negative number (in
double precision a signed zero would flip it without warning). The formula
holds in the physical region only.

Laurent coefficients eps^-2 ... eps^2, by two methods:
  * `laurent_coefficients` (method A1): Cauchy's formula on |eps| = 0.2 with mpmath at
    30 digits, as for the massless box; about 0.7 s per point. Oracle of the tests.
  * `reference_values` (method B): analytic eps expansion of the hypergeometric,
        F(z) = 1 + eps^2 Li2(z) + eps^3 [Li3(z) - S12(z)] + eps^4 [Li4(z) - S22(z) + S13(z)] + O(eps^5),
    with the Nielsen polylogarithms S_{n,p}(z) (real for z < 1) by tanh-sinh quadrature,
    and truncated series algebra; numpy, double precision, about 0.08 ms per point (1e5 points in ~8 s).
    It agrees with A1 to ~1e-14 (|delta| / max(1, |c|)) down to a margin 1e-3
    from the edges of the triangle; closer to the edges it has not been checked.
"""

import functools
import math

import mpmath as mp
import numpy as np

#: Laurent orders, as in Boni: eps^-2 ... eps^2 (N_eps = 5)
ORDERS = (-2, -1, 0, 1, 2)
N_MASTERS = 4
#: method of `reference_values` (enters the cache key)
METHOD = "series"
#: tanh-sinh rule of the Nielsen quadrature: t = 1/(1 + exp(-pi sinh tau)), tau in [-TS_TMAX, TS_TMAX]
#: with step TS_STEP (converged at 1/16: 1/32 gives identical values)
TS_STEP = 1 / 16
TS_TMAX = 4.0
#: points per internal block of `reference_values` (memory: about 10 MB per block)
BLOCK_SIZE = 10_000
#: what identifies the output of `reference_values` (enters the cache key of the datasets: do not change the content)
CACHE_PARAMS = {"orders": list(ORDERS), "method": METHOD, "ts_step": TS_STEP, "ts_tmax": TS_TMAX}
#: how `data.generation.compute_reference` splits the points (never changes the values): method B is numpy, about
#: 0.08 ms per point, so large blocks in one process (more processes would cost more than they save)
COMPUTE_OPTIONS = {"chunk_size": BLOCK_SIZE, "max_workers": 1}
#: method A1 (Cauchy in mpmath); the eps singularities nearest to 0 are at |eps| = 1
CAUCHY_POINTS = 64
CAUCHY_RADIUS = "0.2"
WORKING_DPS = 30
#: precision of the Taylor coefficients of the Gamma-function prefactors
CONSTANTS_DPS = 40
#: the six Nielsen polylogarithms S_{n,p} with n + p <= 4, in the order used below
NIELSEN = ((1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3))


# --------------------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------------------

def _kinematics(s1, s2, s0, lib):
    """Arguments z_i, 1 - z_i (computed directly, no cancellation) and the real logs
    ln(1-x), ln(y-1), ln s2, ln b of the box formula; numpy arrays (lib=np) or mpf (lib=mp)."""
    omz = (s0 / (s2 - s1), s1 / (s2 - s0), s1 * s0 / ((s2 - s1) * (s2 - s0)))
    zs = tuple(1 - v for v in omz)
    log_1mx = lib.log((s2 - s1) / s2)
    log_ym1 = lib.log((s0 - s2) / s2)
    log_s2 = lib.log(s2)
    log_b = lib.log(-s2 / (s0 * s1))
    return zs, omz, log_1mx, log_ym1, log_s2, log_b


def _as_points(points):
    """(N, 2) float64 array of points (s1, s2) inside the open physical region (s0 = 1)."""
    arr = np.asarray(points)
    if np.iscomplexobj(arr):
        raise TypeError("points must be real pairs (s1, s2)")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points must be finite")
    s1, s2 = arr[:, 0], arr[:, 1]
    inside = (s1 < 0) & (s2 > 0) & (1 + s1 - s2 > 0)
    if not inside.all():
        first = arr[~inside][0]
        raise ValueError(f"{int((~inside).sum())} point(s) outside the physical region "
                         f"s1 < 0, s2 > 0, 1 + s1 - s2 > 0 (s0 = 1); first: ({first[0]}, {first[1]})")
    return arr


def _mp_point(s0, s1, s2):
    """The invariants as mpf, checked to lie in the open region (6.38)."""
    s0, s1, s2 = mp.mpf(s0), mp.mpf(s1), mp.mpf(s2)
    if not (s0 > 0 and s1 < 0 and s2 > 0 and s0 + s1 - s2 > 0):
        raise ValueError(f"(s0, s1, s2) = ({s0}, {s1}, {s2}) is outside the physical region "
                         "s0 > 0, s1 < 0, s2 > 0, s0 + s1 - s2 > 0")
    return s0, s1, s2


# --------------------------------------------------------------------------------------
# Method A1: the functions at finite eps (mpmath) and Cauchy's formula
# --------------------------------------------------------------------------------------

def _hyp(eps, z):
    return mp.hyp2f1(-eps, -eps, 1 - eps, z)


def _bubble_constant(eps):
    """Gamma(eps) Gamma(1-eps)^2 / Gamma(2-2eps) (Henn eq. 3.17 without the power)."""
    return mp.gamma(eps) * mp.gamma(1 - eps) ** 2 / mp.gamma(2 - 2 * eps)


def box(s0, s1, s2, eps):
    """G_1111 = BDK I_4^{1m} (4.26) at (s, t, m^2) = (s0, s1, s2) of the region, by the formula of the module docstring.

    mpmath, any complex eps with 0 < |eps| < 1; the side of the cut is the explicit e^{i pi eps}.
    """
    s0, s1, s2 = _mp_point(s0, s1, s2)
    eps = mp.mpmathify(eps)
    (z1, z2, z3), _, log_1mx, log_ym1, _, log_b = _kinematics(s1, s2, s0, mp)
    r_g = mp.gamma(1 + eps) * mp.gamma(1 - eps) ** 2 / mp.gamma(1 - 2 * eps)
    power_x = mp.exp(eps * log_1mx)
    power_y = mp.exp(eps * mp.mpc(log_ym1, mp.pi))          # (1 - y + i0)^eps = e^{i pi eps} (y-1)^eps
    bracket = power_x * _hyp(eps, z1) + power_y * (_hyp(eps, z2) - power_x * _hyp(eps, z3))
    return 2 * r_g / (eps**2 * s0 * s1) * mp.exp(eps * log_b) * bracket


def boni_basis(s1, s2, eps, s0=1):
    """(M1, M2, M3, M4) of the basis at finite complex eps (mpmath): box, bubble(t), bubble(m^2), bubble(s)."""
    s0, s1, s2 = _mp_point(s0, s1, s2)
    eps = mp.mpmathify(eps)
    norm = mp.exp(eps * mp.euler)
    cb = _bubble_constant(eps)
    log_s2 = mp.log(s2)
    bub_t = mp.exp(-eps * mp.log(-s1)) * cb                       # s1 < 0: no phase
    bub_m = mp.exp(-eps * mp.mpc(log_s2, -mp.pi)) * cb             # (-s2 - i0)^{-eps}
    bub_s = mp.exp(-eps * mp.mpc(mp.log(s0), -mp.pi)) * cb         # (-s0 - i0)^{-eps}
    return [
        norm * s1 * mp.exp((1 + eps) * log_s2) * box(s0, s1, s2, eps),
        norm * mp.exp((1 + eps) * log_s2) / s0 * bub_t,
        norm * mp.exp(eps * log_s2) * bub_m,
        norm * s1 * mp.exp((eps - 1) * log_s2) * bub_s,
    ]


def laurent_coefficients(point, orders=ORDERS, s0=1):
    """Method A1: complex Laurent coefficients at one point (s1, s2), shape (4, len(orders)).

    Cauchy's formula on |eps| = CAUCHY_RADIUS with CAUCHY_POINTS nodes at WORKING_DPS digits
    (trapezoidal rule, exponentially convergent: the function is analytic for 0 < |eps| < 1).
    `s0` != 1 is only for the scale-invariance check: the result depends on s1/s0, s2/s0.
    """
    arr = np.asarray(point)
    if np.iscomplexobj(arr):
        raise TypeError("point must be a real pair (s1, s2)")
    s1, s2 = (float(v) for v in np.asarray(arr, dtype=np.float64).reshape(2))
    with mp.workdps(WORKING_DPS):
        s0m, s1m, s2m = _mp_point(s0, s1, s2)
        radius = mp.mpf(CAUCHY_RADIUS)
        nodes = [radius * mp.expjpi(2 * mp.mpf(j) / CAUCHY_POINTS) for j in range(CAUCHY_POINTS)]
        values = [boni_basis(s1m, s2m, e, s0m) for e in nodes]
        out = np.empty((N_MASTERS, len(orders)), dtype=complex)
        for i in range(N_MASTERS):
            for k, order in enumerate(orders):
                c = mp.fsum(v[i] * e ** (-order) for v, e in zip(values, nodes)) / CAUCHY_POINTS
                out[i, k] = complex(c)
    return out


# --------------------------------------------------------------------------------------
# Method B: eps expansion in Nielsen polylogarithms (numpy)
# --------------------------------------------------------------------------------------

def _frozen(arr):
    arr.flags.writeable = False
    return arr


@functools.lru_cache(maxsize=None)
def _series_constants():
    """Taylor coefficients at eps = 0 of e^{eps gamma_E} r_G (orders 0..4) and of
    e^{eps gamma_E} eps Gamma(eps) Gamma(1-eps)^2 / Gamma(2-2eps) (orders 0..3); computed once."""
    with mp.workdps(CONSTANTS_DPS):
        common = lambda e: mp.exp(e * mp.euler) * mp.gamma(1 + e) * mp.gamma(1 - e) ** 2
        r_g = mp.taylor(lambda e: common(e) * mp.rgamma(1 - 2 * e), 0, 4)
        bub = mp.taylor(lambda e: common(e) * mp.rgamma(2 - 2 * e), 0, 3)
        return (_frozen(np.array([float(v) for v in r_g])),
                _frozen(np.array([float(v) for v in bub])))


@functools.lru_cache(maxsize=None)
def _tanh_sinh_rule(step, tmax):
    """Nodes t, 1 - t, ln t and the weights of dt/t on [0, 1], all without cancellation."""
    tau = np.arange(-tmax, tmax + step / 2, step)
    q = np.exp(-np.pi * np.sinh(tau))            # t = 1/(1 + q)
    t = 1 / (1 + q)
    one_minus_t = q / (1 + q)
    log_t = -np.log1p(q)
    weight_over_t = step * np.pi * np.cosh(tau) * q / (1 + q)
    return tuple(_frozen(a) for a in (t, one_minus_t, log_t, weight_over_t))


def _log_one_minus_zt(z, omz, t, omt):
    """ln(1 - z t) for real z < 1 and t in [0, 1]; near t = 1 as ln[(1 - z) + z (1 - t)]."""
    small = t < 0.5
    near_zero = np.log1p(-np.where(small, z * t, 0.0))
    near_one = np.log(np.where(small, 1.0, omz + z * omt))
    return np.where(small, near_zero, near_one)


def nielsen_polylogs(z, one_minus_z=None):
    """The six Nielsen polylogarithms S_{n,p}(z) of NIELSEN for real z < 1, shape (len(z), 6).

    S_{n,p}(z) = (-1)^{n+p-1} / ((n-1)! p!) int_0^1 ln^{n-1}(t) ln^p(1 - z t) dt/t, by the
    tanh-sinh rule (TS_STEP, TS_TMAX). `one_minus_z` avoids the cancellation in 1 - z.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    omz = 1 - z if one_minus_z is None else np.asarray(one_minus_z, dtype=np.float64).reshape(-1)
    if not np.all(omz > 0):
        raise ValueError("the expansion holds for real z < 1")
    t, omt, log_t, w = _tanh_sinh_rule(TS_STEP, TS_TMAX)
    big_l = _log_one_minus_zt(z[:, None], omz[:, None], t[None, :], omt[None, :])
    out = np.empty((z.size, len(NIELSEN)))
    for k, (n, p) in enumerate(NIELSEN):
        c = (-1) ** (n + p - 1) / (math.factorial(n - 1) * math.factorial(p))
        out[:, k] = c * ((log_t ** (n - 1) * w)[None, :] * big_l**p).sum(axis=1)
    return out


def hypergeometric_series(z, one_minus_z=None):
    """Taylor coefficients eps^0 ... eps^4 of 2F1(-eps, -eps; 1-eps; z), real z < 1, in Nielsen polylogarithms.

    Shape (len(z), 5): [1, 0, Li2, Li3 - S12, Li4 - S22 + S13] (Li_n = S_{n-1,1}).
    """
    s = nielsen_polylogs(z, one_minus_z)
    li2, li3, s12, li4, s22, s13 = (s[:, k] for k in range(len(NIELSEN)))
    return np.stack([np.ones_like(li2), np.zeros_like(li2), li2, li3 - s12, li4 - s22 + s13], axis=-1)


def _series_exp(v, k=5):
    """Taylor coefficients 0..k-1 of exp(eps v), for an array v: shape v.shape + (k,)."""
    v = np.asarray(v)
    out = np.empty(v.shape + (k,), dtype=complex)
    out[..., 0] = 1
    for j in range(1, k):
        out[..., j] = out[..., j - 1] * v / j
    return out


def _series_mul(a, b):
    """Product of truncated series along the last axis (same length)."""
    k = a.shape[-1]
    out = np.zeros(np.broadcast(a, b).shape, dtype=complex)
    for i in range(k):
        out[..., i:] += a[..., i:i + 1] * b[..., :k - i]
    return out


def _series_block(s1, s2):
    """Method B on one block of points (s0 = 1): complex array (len(s1), 4, 5), orders eps^-2..eps^2."""
    s0 = 1.0
    zs, omzs, log_1mx, log_ym1, log_s2, log_b = _kinematics(s1, s2, s0, np)
    f1, f2, f3 = (hypergeometric_series(z, omz) for z, omz in zip(zs, omzs))
    r_g, bub = _series_constants()
    n = len(s1)
    power_x = _series_exp(log_1mx)
    power_y = _series_exp(log_ym1 + 1j * np.pi)     # explicit side of the cut: e^{i pi eps}
    bracket = _series_mul(power_x, f1) + _series_mul(power_y, f2 - _series_mul(power_x, f3))
    prefactor = _series_mul(np.broadcast_to(r_g, bracket.shape), _series_exp(log_s2 + log_b))
    bub = np.broadcast_to(bub, (n, 4))
    out = np.zeros((n, N_MASTERS, len(ORDERS)), dtype=complex)
    # box: 2 (s2/s0) e^{eps gamma_E} r_G eps^-2 (s2 b)^eps [bracket]
    out[:, 0, :] = (2 * s2 / s0)[:, None] * _series_mul(prefactor, bracket)
    # bubbles: e^{eps gamma_E} Gamma(eps)... starts at eps^-1
    out[:, 1, 1:] = (s2 / s0)[:, None] * _series_mul(_series_exp(log_s2 - np.log(-s1), 4), bub)
    out[:, 2, 1:] = _series_mul(_series_exp(np.full(n, 1j * np.pi), 4), bub)
    out[:, 3, 1:] = (s1 / s2)[:, None] * _series_mul(_series_exp(log_s2 - np.log(s0) + 1j * np.pi, 4), bub)
    return out


def reference_values(points, orders=ORDERS):
    """Method B: Laurent coefficients at points (s1, s2) (s0 = 1), complex array (N, 4, len(orders)).

    `points` has shape (N, 2), real, inside the open triangle s1 < 0, s2 > 0, 1 + s1 - s2 > 0
    (ValueError otherwise; TypeError for complex input). Masters in the order of the
    matrices: box, bubble(t), bubble(m^2), bubble(s). Real and imaginary parts are split
    downstream. Computed in blocks of BLOCK_SIZE points; about 0.08 ms per point (1e5 points in ~8 s).
    """
    idx = [ORDERS.index(o) for o in orders]          # ValueError for an order not computed
    arr = _as_points(points)
    out = np.empty((len(arr), N_MASTERS, len(ORDERS)), dtype=complex)
    for start in range(0, len(arr), BLOCK_SIZE):
        block = arr[start:start + BLOCK_SIZE]
        out[start:start + len(block)] = _series_block(block[:, 0], block[:, 1])
    return out[:, :, idx]
