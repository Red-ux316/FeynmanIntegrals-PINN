"""One-loop massless box: differential-equation system in Boni's basis.

Basis (Boni, thesis eq. 6.26):  M = (t^eps I_{0101}, t^eps I_{1010}, t^{2+eps} I_{1111}),
with the integrals of the (s,t) box (CMZ eq. 2.4-2.6; not the propagators printed in Boni eq. 6.23,
which do not reproduce the matrix A.2).

System (Boni eq. A.2, in the variable x = t/s):   dM/dx = A(x, eps) M,
    A = A0(x) + eps * A1(x)          (linear in eps)

Variables: the thesis trains in x_B = c_x * t/s with c_x = 10 (read off
Boni's figures); A.2 is written for c_x = 1. We work in x = t/s and convert
with `to_boni_x` / `from_boni_x`. The physical s-channel region is -1 < x < 0;
Boni's training domain x_B in (-9, -1) is x in (-0.9, -0.1).
"""

import sympy as sp

EPS, X = sp.symbols("epsilon x")

#: c_x of Boni's variable x_B = c_x t/s
C_X_BONI = 10
#: Boni's training domain in x = t/s
BONI_DOMAIN = (-0.9, -0.1)
#: singular points of the system (x = 0: t = 0, physical; x = -1: u = 0, spurious)
SINGULAR_POINTS = (0, -1)
N_MASTERS = 3
#: the variables of the system (common interface of `pinn_feynman.systems`): one, x = t/s
VARIABLES = ("x",)
#: name of the problem in figure titles, and axis labels of the figures (in Boni's x_B, the network variable)
TITLE = "Massless box"
VARIABLE_LABELS = (r"$x_B = 10\,t/s$",)


def connection_matrix():
    """A(x, eps) of Boni eq. (A.2), as a sympy Matrix in (X, EPS)."""
    x, e = X, EPS
    return sp.Matrix([
        [0, 0, 0],
        [0, e / x, 0],
        [2 * (1 - 2 * e) / (1 + x), 2 * (-1 + 2 * e) * x / (1 + x), (1 + x + e * x) / (x * (1 + x))],
    ])


def connection_matrix_orders(variable="x"):
    """(A0, A1) with A = A0 + eps A1 (sympy Matrices in X); `variable` can only be 'x' (common interface)."""
    if variable not in VARIABLES:
        raise ValueError(f"unknown variable {variable!r}; available: {list(VARIABLES)}")
    a = connection_matrix().applyfunc(sp.expand)
    a0 = a.subs(EPS, 0)
    a1 = a.applyfunc(lambda entry: sp.diff(entry, EPS))
    if sp.simplify(a - a0 - EPS * a1) != sp.zeros(N_MASTERS, N_MASTERS):
        raise ValueError("connection matrix is not linear in eps")
    return a0.applyfunc(sp.simplify), a1.applyfunc(sp.simplify)


def numeric_orders(modules="numpy"):
    """Callables x -> A0(x), x -> A1(x) (nested lists), via sympy.lambdify."""
    a0, a1 = connection_matrix_orders()
    return sp.lambdify(X, a0, modules=modules), sp.lambdify(X, a1, modules=modules)


def to_boni_x(x):
    """x = t/s  ->  Boni's x_B = c_x t/s."""
    return C_X_BONI * x


def from_boni_x(x_boni):
    """Boni's x_B = c_x t/s  ->  x = t/s."""
    return x_boni / C_X_BONI
