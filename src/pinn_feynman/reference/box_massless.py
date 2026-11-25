"""Exact reference values of the one-loop massless box master integrals.

Ground truth for training (boundary points) and testing, replacing AMFlow, which
Boni used (thesis sec. 6.1.1). All formulas are literature (a); their assembly and
the conventions are documented in this folder's README.md, with the checks made.

Integrals (measure and propagators as Henn eq. 3.6: d^Dk/(i pi^{D/2}), [-(k+q)^2]):
  * bubble, exact in eps (Henn eq. 3.17 with a1 = a2 = 1):
        G_bub(p^2) = (-p^2)^(-eps) Gamma(eps) Gamma(1-eps)^2 / Gamma(2-2eps)
  * box, exact in eps: from BDK (hep-ph/9306240) eq. (4.18) with (4.12), (2.15), (2.11),
        I_box(s,t) = 2 r_G/(eps^2 s t) [ (-s)^(-eps) 2F1(1,-eps;1-eps;1+s/t)
                                         + (-t)^(-eps) 2F1(1,-eps;1-eps;1+t/s) ]
    with r_G = Gamma(1+eps) Gamma(1-eps)^2 / Gamma(1-2eps) (BDK after eq. I.4).
Analytic continuation (BDK eq. 4.24): log(-s) -> ln|s| - i pi for s > 0.
The combined box formula is written in the variables of the physical s-channel
region (1+t/s in (0,1), 1+s/t < 0: no hypergeometric branch cut is crossed).

Boni's basis and conventions (fixed against his Fig. 6.5-6.7):
    M = e^{eps gamma_E} (t^eps G_{0101}, t^eps G_{1010}, t^{2+eps} G_{1111}),
    t^eps = exp(eps log t) with the principal log (log t = ln|t| + i pi for t < 0).
"""

import mpmath as mp
import numpy as np

#: Laurent orders used by Boni: eps^-2 ... eps^2 (N_eps = 5)
ORDERS = (-2, -1, 0, 1, 2)
#: Cauchy contour for Laurent coefficients (nearest eps-singularity is at |eps| = 1)
CAUCHY_POINTS = 64
CAUCHY_RADIUS = mp.mpf("0.2")
WORKING_DPS = 30
#: what identifies the output of `reference_values` (enters the cache key of the datasets: do not change the content)
CACHE_PARAMS = {"orders": list(ORDERS), "cauchy_points": CAUCHY_POINTS, "cauchy_radius": str(CAUCHY_RADIUS),
                "working_dps": WORKING_DPS}
#: how `data.generation.compute_reference` splits the points (never changes the values): small chunks over many
#: processes, because the evaluation is mpmath, about 0.1 s per point
COMPUTE_OPTIONS = {"chunk_size": 10, "max_workers": None}


def log_minus(v):
    """log(-v - i0) with the Feynman prescription (BDK eq. 4.24).

    Real v > 0 gives ln v - i pi; for complex v with Im v > 0 the principal
    branch of log(-v) is its continuous continuation.
    """
    v = mp.mpc(v)
    if v.imag == 0 and v.real > 0:
        return mp.log(v.real) - 1j * mp.pi
    return mp.log(-v)


def r_gamma(eps):
    return mp.gamma(1 + eps) * mp.gamma(1 - eps) ** 2 / mp.gamma(1 - 2 * eps)


def bubble(p2, eps):
    """G_{1,0,1,0}-type massless bubble with external invariant p2 (Henn eq. 3.17)."""
    return mp.exp(-eps * log_minus(p2)) * mp.gamma(eps) * mp.gamma(1 - eps) ** 2 / mp.gamma(2 - 2 * eps)


def box(s, t, eps):
    """G_{1,1,1,1}: massless box, exact in eps (BDK eqs. 4.18, 4.12, 2.15, 2.11)."""
    f = lambda z: mp.hyp2f1(1, -eps, 1 - eps, z)
    return 2 * r_gamma(eps) / (eps**2 * s * t) * (
        mp.exp(-eps * log_minus(s)) * f(1 + s / t) + mp.exp(-eps * log_minus(t)) * f(1 + t / s)
    )


def boni_basis(x, eps, s=1):
    """(M1, M2, M3) at x = t/s for finite complex eps (Boni's basis and conventions)."""
    s = mp.mpf(s)
    t = mp.mpf(x) * s
    t_eps = mp.exp(eps * mp.log(t))                 # principal log: ln|t| + i pi for t < 0
    norm = mp.exp(eps * mp.euler)
    return (
        norm * t_eps * bubble(t, eps),
        norm * t_eps * bubble(s, eps),
        norm * t**2 * t_eps * box(s, t, eps),
    )


def cauchy_nodes():
    """The CAUCHY_POINTS nodes eps_j on the circle |eps| = CAUCHY_RADIUS (at the current mpmath precision)."""
    return [CAUCHY_RADIUS * mp.expjpi(2 * mp.mpf(j) / CAUCHY_POINTS) for j in range(CAUCHY_POINTS)]


def cauchy_coefficients(values, nodes, orders=ORDERS):
    """Coefficients of eps^k, k in `orders`, of a function given by its values at `nodes` (mpmath numbers).

    Cauchy's formula with the trapezoidal rule: c_k = (1/N) sum_j f(eps_j) eps_j^(-k), exponentially convergent for
    functions analytic in an annulus around the circle.
    """
    return [mp.fsum(v * e ** (-order) for v, e in zip(values, nodes)) / len(nodes) for order in orders]


def laurent_coefficients(x, orders=ORDERS):
    """Complex Laurent coefficients M_i^{(l)}(x), shape (3, len(orders)), from `cauchy_coefficients`."""
    with mp.workdps(WORKING_DPS):
        nodes = cauchy_nodes()
        values = [boni_basis(x, e) for e in nodes]
        out = np.empty((3, len(orders)), dtype=complex)
        for i in range(3):
            out[i] = [complex(c) for c in cauchy_coefficients([v[i] for v in values], nodes, orders)]
    return out


def reference_values(xs, orders=ORDERS):
    """Laurent coefficients on a grid of x = t/s: complex array of shape (len(xs), 3, len(orders)).

    Real and imaginary parts are the targets of Boni's two separate trainings.
    Cost: about 0.2 s per point.
    """
    return np.stack([laurent_coefficients(x, orders) for x in np.atleast_1d(xs)])
