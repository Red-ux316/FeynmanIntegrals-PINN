"""Non-planar two-loop three-point function with a massive loop (Boni sec. 6.3.3): the two masters of the top sector.

Basis (Boni eq. 6.57 = von Manteuffel-Tancredi arXiv:1701.05905 eq. 2.3, "vMT"), with x = -s/m^2:
    m10 = eps^4 s^2 I_{1,1,1,1,1,1,0},    m11 = eps^4 s^2 (s + 16 m^2) / (2 (1 + 2 eps)) I_{1,2,1,1,1,1,0}.
Both are finite: their Taylor series in eps starts at eps^4 (vMT sec. 4.1); the network learns eps^0 .. eps^4.

System (vMT eq. 2.7-2.8; Boni eq. 6.58 with the matrix A.12):
    dm/dx = [B(x) + eps D(x)] m + N(eps, x),    N = (0, N11),
the inhomogeneous term N comes from the nine masters of the subtopologies and is known in closed form: its only
non-zero coefficient is N11^(4) (vMT eq. 4.2, continued in eqs. 4.33-4.36). It is supplied by the forcing provider
`providers.nonplanar_three_point` (bottom-up formulation): this module holds the 2 x 2 matrix only.

Boni's (A.12) is equal to B + eps D except for the eps term of the entry (2, 1): Boni prints (1 + eps)/(2x), vMT
2 eps/x. The matrix here is vMT's. The difference only acts on the orders
>= eps^5 (the D term couples order n to n - 1, and m^(n) = 0 for n < 4), so it changes nothing in eps^0 .. eps^4.

Singular points of the matrix: x = 0 (residue [[0, -1/2], [1/2, 1]], double exponent 1/2 with a Jordan block),
x = 16 and infinity. x = -4 (threshold s = 4 m^2) is a square-root branch point of the forcing term, not of the
matrix. x = 4 (pseudo-threshold, listed by Boni (6.59)) is not a branch point of N11^(4): the roots of the single
subtopologies cancel. At x = 0 (massless cut, s = 0) the forcing has a logarithm, not a root; the point is
already a singular point of the matrix.
"""

import sympy as sp

EPS = sp.Symbol("epsilon")
X = sp.Symbol("x")
#: singular points of B + eps D (the threshold x = -4 is a branch point of the forcing term only)
SINGULAR_POINTS = (0, 16)
#: branch points of the forcing term N11^(4) on the real axis that are not already in SINGULAR_POINTS: only the
#: threshold x = -4 (sqrt(x + 4)). x = 4 is not one (the roots cancel in N11^(4)); at x = 0 the forcing has a log, and
#: the point is a pole of the matrix anyway
FORCING_BRANCH_POINTS = (-4,)
N_MASTERS = 2
MASTER_NAMES = ("m10", "m11")
VARIABLES = ("x",)
#: Boni (6.60): -20 <= x <= c_x = 0.1, taken literally (x = 0 is inside the domain)
DEFAULT_DOMAIN = (-20.0, 0.1)
#: orders drawn by `evaluation.evaluate` in the solution curves: eps^4 only, the others vanish (as Boni's Fig. 6.22)
SHOWN_ORDERS = (4,)
#: orders of eps that the physics does not set to zero (vMT sec. 4.1: m10, m11 start at eps^4). kappa_U of the reports is
#: built on these only: kappa = ||B||_F / sqrt(2), without the eps D term
NONZERO_ORDERS = (4,)
TITLE = "Non-planar three-point"
VARIABLE_LABELS = (r"$x = -s/m^2$",)
FIGURE_LABELS = {
    "variable": r"$x = -s/m^2$",
    "along": r"$x = -s/m^2$   (threshold of the forcing term at $x = -4$, singular point $x = 0$)",
    "distance": r"distance from the singular point $x = 0$, $|x|$",
}


def homogeneous_matrices():
    """(B, D) of vMT eq. (2.8), sympy Matrices in X."""
    x = X
    b = sp.Matrix([[0, 1 / (2 * (x - 16)) - 1 / (2 * x)],
                   [1 / (2 * x), 1 / x]])
    d = sp.Matrix([[-2 / x, 1 / (x - 16) - 1 / x],
                   [2 / x, 1 / x - 1 / (x - 16)]])
    return b, d


def connection_matrix():
    """A(x, eps) = B + eps D (rows = derivatives)."""
    b, d = homogeneous_matrices()
    return b + EPS * d


def connection_matrix_orders(variable="x"):
    """(A0, A1) = (B, D); `variable` can only be 'x'."""
    if variable not in VARIABLES:
        raise ValueError(f"unknown variable {variable!r}; available: {list(VARIABLES)}")
    return homogeneous_matrices()


def numeric_orders(modules="numpy"):
    """Callables x -> B(x), D(x) (nested lists), via sympy.lambdify."""
    return tuple(sp.lambdify(X, a, modules=modules) for a in connection_matrix_orders())
