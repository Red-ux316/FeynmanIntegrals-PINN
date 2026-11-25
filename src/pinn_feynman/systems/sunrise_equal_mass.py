"""Two-loop equal-mass sunrise: differential-equation system in Boni's basis, m^2 = 3.

Basis (Boni, thesis eq. 6.50), raw integrals of the family (6.48)-(6.49), m^2 = 3, mu = 1:
    M = (I_11000, I_11100, I_21100) = (tadpole^2, sunrise, sunrise with a squared propagator),
literal propagators q^2 - m^2 (Minkowski, +i0), measure d^dk/(i pi^{d/2}) per loop, no exp(eps gamma_E).

System (Boni eq. A.10, in the variable s = p^2):   dM/ds = A(s, eps) M,
    A = A0(s) + eps A1(s) + eps^2 A2(s)          (quadratic in eps, no eps in a denominator)
Row 1 vanishes (the tadpole is constant), row 2 is the homogeneity relation.

Singular points: s = 0, 3 (pseudo-threshold, (m1 + m2 - m3)^2 = m^2) and 27 (threshold, 9 m^2); s = 9, cited by the
thesis (6.52), is not a singularity of (A.10).
"""

import sympy as sp

EPS = sp.Symbol("epsilon")
S = sp.Symbol("s")
#: the variable symbol under the name used by `systems.analysis` (one-variable systems)
X = S
#: m^2 of Boni's sunrise (thesis sec. 6.3.2)
MASS_SQUARED = 3
#: singular points of (A.10): s = 0, the pseudo-threshold m^2 = 3 and the threshold 9 m^2 = 27
SINGULAR_POINTS = (0, 3, 27)
N_MASTERS = 3
MASTER_NAMES = ("tadpole_squared", "sunrise", "sunrise_dotted")
#: the variables of the system (common interface of `pinn_feynman.systems`): one, s = p^2
VARIABLES = ("s",)
#: name of the problem in figure titles, and axis labels of the figures
TITLE = "Equal-mass sunrise"
VARIABLE_LABELS = (r"$s = p^2$  ($m^2 = 3$)",)
#: axis labels of the one-variable figures of `evaluation.evaluate` (the massless box keeps its x_B labels)
FIGURE_LABELS = {
    "variable": r"$s = p^2$",
    "along": r"$s = p^2$   (singular points of (A.10) at $s = 0, 3, 27$)",
    "distance": r"distance from the singular point $s = 0$, $|s|$",
}


def connection_matrix():
    """A(s, eps) of Boni eq. (A.10), as a sympy Matrix in (S, EPS) (rows = derivatives)."""
    s, e = S, EPS
    den3 = 3 * (s - 27) * (s - 3)
    den = (s - 27) * (s - 3) * s
    a31 = (-2 + 4 * e - 2 * e**2) / den3
    a32 = (-2 + 7 * e - 6 * e**2) * (s - 9) / den
    a33 = (-(162 - 30 * s) - e * (s**2 + 30 * s - 243)) / den
    return sp.Matrix([
        [0, 0, 0],
        [0, (1 - 2 * e) / s, -sp.Integer(9) / s],
        [a31, a32, a33],
    ])


def connection_matrix_orders(variable="s"):
    """(A0, A1, A2) with A = A0 + eps A1 + eps^2 A2 (sympy Matrices in S); `variable` can only be 's'.

    Error if A is not a polynomial of degree <= 2 in eps.
    """
    if variable not in VARIABLES:
        raise ValueError(f"unknown variable {variable!r}; available: {list(VARIABLES)}")
    a = connection_matrix().applyfunc(sp.expand)
    orders = tuple(a.applyfunc(lambda entry, j=j: sp.cancel(sp.expand(entry).coeff(EPS, j))) for j in range(3))
    rebuilt = orders[0] + EPS * orders[1] + EPS**2 * orders[2]
    if (a - rebuilt).applyfunc(sp.cancel) != sp.zeros(N_MASTERS, N_MASTERS):
        raise ValueError("connection matrix is not quadratic in eps")
    return orders


def numeric_orders(modules="numpy"):
    """Callables s -> A0(s), A1(s), A2(s) (nested lists), via sympy.lambdify."""
    return tuple(sp.lambdify(S, a, modules=modules) for a in connection_matrix_orders())
