"""One-loop one-mass box: differential-equation system in Boni's basis, physical region.

Basis, in the order of Boni's matrices:
    M = (s1 s2^{1+eps} I_1111,  s0^-1 s2^{1+eps} I_0101,  s2^eps I_1001,  s1 s2^{-1+eps} I_1010)
      = (box, bubble in t, bubble in m^2, bubble in s),
with s0 = s, s1 = t, s2 = m^2 (Boni A.3), with the propagators of BDK.

System:  dM/ds_k = A_{s_k}(s0, s1, s2, eps) M,  k = 0, 1, 2. A_{s1}, A_{s2} are Boni's (A.4)-(A.5)
with typographical corrections (two entries of A.4, one of A.5, the missing fourth row of A.4),
derived independently from BDK eq. (4.33);
A_{s0} (not in the thesis) follows from the scale relation below. They are linear in eps,
A = A^(0) + eps A^(1), real for real invariants, pairwise integrable, and
s0 A_{s0} + s1 A_{s1} + s2 A_{s2} = 0 (scale invariance of the basis with mu = 1).
Their denominators factor into s0, s1, s2, s0 + s1 - s2, s0 - s2, s1 - s2 (Boni eq. 6.37).

Variables: s0 = S0_VALUE = 1 (the basis depends on s1/s0, s2/s0 only); the network works on
(s1, s2). Physical region (Boni 6.38): the open triangle s1 < 0, s2 > 0, 1 + s1 - s2 > 0,
whose edges are t = 0, m^2 = 0 and u = 0. The training domain is a triangle with a margin
from the edges (DEFAULT_MARGIN = 0.1 by default), its vertices are the boundary points.
This system has two variables: it is not registered in `analysis.SYSTEM_MODULES` (one variable).
"""

import numpy as np
import sympy as sp

EPS, S0, S1, S2 = sp.symbols("epsilon s0 s1 s2")
#: invariant name -> sympy symbol
INVARIANTS = {"s0": S0, "s1": S1, "s2": S2}
#: the variables of the network (s0 is fixed to S0_VALUE)
VARIABLES = ("s1", "s2")
S0_VALUE = 1
N_MASTERS = 4
MASTER_NAMES = ("box", "bubble_t", "bubble_m2", "bubble_s")
#: name of the problem in figure titles, and axis labels of the figures (physical variables, s0 = 1)
TITLE = "One-mass box"
VARIABLE_LABELS = (r"$s_1 = t/s$", r"$s_2 = m^2/s$")
#: the edges of the physical region, in the order of the columns of `edge_distances`
EDGE_LABELS = (r"$t = 0$", r"$m^2 = 0$", r"$u = 0$")
#: margin of the current training triangle from the edges of the region
DEFAULT_MARGIN = 0.1
#: absolute tolerance of `in_domain` on the margin (rounding of the vertices)
MARGIN_TOLERANCE = 1e-12
#: the factors of the denominators of the matrices (Boni eq. 6.37)
DENOMINATOR_FACTORS = (S0, S1, S2, S0 + S1 - S2, S0 - S2, S1 - S2)


def connection_matrix(variable):
    """A_{s_k}(s0, s1, s2, eps) for variable 's0', 's1' or 's2' (sympy Matrix)."""
    s0, s1, s2, e = S0, S1, S2, EPS
    c = 1 - 2 * e
    w = s0 + s1 - s2                       # w = -u > 0 in the region
    if variable == "s1":
        return sp.Matrix([
            [e * (s2 - s0) / (s1 * w), 2 * c * s0 / ((s1 - s2) * w), -2 * c * s2 / ((s1 - s2) * w),
             2 * c * s2**2 / (s0 * s1 * (s2 - s0 - s1))],
            [0, -e / s1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1 / s1],
        ])
    if variable == "s2":
        return sp.Matrix([
            [(w + e * (s0 + s1 - 2 * s2)) / (s2 * w), -2 * c * s0 / ((s1 - s2) * w),
             -2 * c * s1 * (s0 + s1 - 2 * s2) / ((s2 - s0) * (s1 - s2) * w),
             -2 * c * s2**2 / (s0 * (s2 - s0) * (s2 - s0 - s1))],
            [0, (1 + e) / s2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, (e - 1) / s2],
        ])
    if variable == "s0":
        return sp.Matrix([
            [-(w + e * (s1 - s2)) / (s0 * w), -2 * c / w, -2 * c * s1 * s2 / (s0 * (s0 - s2) * w),
             2 * c * s2**2 / (s0 * (s0 - s2) * w)],
            [0, -1 / s0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -e / s0],
        ])
    raise ValueError(f"unknown variable {variable!r}; available: {sorted(INVARIANTS)}")


def connection_matrix_orders(variable, s0=S0_VALUE):
    """(A0, A1) with A_{s_k} = A0 + eps A1 (sympy Matrices); error if not linear in eps.

    `s0`: value substituted for s0 (default S0_VALUE = 1); None keeps it symbolic.
    """
    a = connection_matrix(variable)
    if s0 is not None:
        a = a.subs(S0, s0)
    a0 = a.subs(EPS, 0)
    a1 = a.diff(EPS)
    if (a - a0 - EPS * a1).applyfunc(sp.simplify) != sp.zeros(N_MASTERS, N_MASTERS):
        raise ValueError(f"connection matrix A_{variable} is not linear in eps")
    return a0.applyfunc(sp.simplify), a1.applyfunc(sp.simplify)


def numeric_orders(modules="numpy"):
    """{'s1': (f0, f1), 's2': (f0, f1)}: callables (s1, s2) -> A0, A1 at s0 = S0_VALUE (nested lists)."""
    out = {}
    for variable in VARIABLES:
        a0, a1 = connection_matrix_orders(variable)
        out[variable] = (sp.lambdify((S1, S2), a0, modules=modules),
                         sp.lambdify((S1, S2), a1, modules=modules))
    return out


def edge_distance(points):
    """min(-s1, s2, 1 + s1 - s2) for points (..., 2) = (s1, s2), s0 = 1: positive inside the region,
    equal to the margin of the smallest triangle T_margin containing the point."""
    p = np.asarray(points, dtype=np.float64)
    s1, s2 = p[..., 0], p[..., 1]
    return np.minimum(np.minimum(-s1, s2), 1 + s1 - s2)


def edge_distances(points):
    """(-s1, s2, 1 + s1 - s2) for points (..., 2) = (s1, s2), s0 = 1, shape (..., 3): the distances (in the
    sense of `edge_distance`) from the edges t = 0, m^2 = 0, u = 0 (EDGE_LABELS); their minimum is `edge_distance`."""
    p = np.asarray(points, dtype=np.float64)
    s1, s2 = p[..., 0], p[..., 1]
    return np.stack([-s1, s2, 1 + s1 - s2], axis=-1)


def in_domain(points, margin=0.0):
    """True where the point lies in the open region (margin = 0) or in the closed triangle
    T_margin = {s1 <= -margin, s2 >= margin, 1 + s1 - s2 >= margin} (margin > 0).

    The closed triangle is tested up to MARGIN_TOLERANCE: in floating point the vertex
    (2m - 1, m) of `vertices(m)` has 1 + s1 - s2 slightly below m (0.09999999999999998 for m = 0.1).
    """
    d = edge_distance(points)
    return (d > 0) & (d >= margin - MARGIN_TOLERANCE)


def vertices(margin=DEFAULT_MARGIN):
    """The three vertices (s1, s2) of T_margin, shape (3, 2): near t = m^2 = 0, near m^2 = u = 0,
    near t = u = 0: the boundary points."""
    if not 0 < margin < 1 / 3:
        raise ValueError("the margin must be in (0, 1/3)")
    m = float(margin)
    return np.array([[-m, m], [2 * m - 1, m], [-m, 1 - 2 * m]])
