"""Frobenius expansion of the massless box around x = 0, in Boni's basis.

Basis and conventions: M = e^{eps gE} (t^eps G_0101, t^eps G_1010, t^{2+eps} G_1111), x = t/s with s = 1 (physical
s-channel for -1 < x < 0), log t principal (ln|t| + i pi for t < 0). The system is dM/dx = A(x) M with A from
Boni (A.2); beta0 = 2(1 - 2 eps).

The balance T = diag(1, 1, -x) turns the system into x dM~/dx = (Lambda + x A~_-1 / (1 + x)) M~, whose local solution
at x = 0 is M~ = U~(x) x^Lambda d. Back in Boni's basis, M = H(x) x^Lambda d with H = diag(1, 1, -x) U~(x).

The four steps of the computation:
  1. `frobenius_series`: Lambda and the first n coefficients of U~, from the recursion
     U_j (Lambda + j) - Lambda U_j = sum_{k=1..j} A_k U_{j-k};
  2. `fix_constants`: the constants d from H(x0) x0^Lambda d = M(x0), with the exact M;
  3. `local_m3`: the third row of H(x) x^Lambda d and its derivative, with ln x = ln(-x) + i pi (the whole function)
     or ln x = i pi (the analytic part);
  4. `exact_m3`: the exact M_3 and its derivative, the third row of A(x) M(x).

The exact box, the matrix A and Cauchy's formula for the orders in eps come from the package
(`reference.box_massless`, `systems.box_massless`); `parts` puts the four steps together for the figure of the analytic
and singular parts.
"""

import mpmath as mp
import sympy as sp

from pinn_feynman.reference.box_massless import boni_basis, cauchy_coefficients, cauchy_nodes
from pinn_feynman.systems.box_massless import EPS, X, connection_matrix

mp.mp.dps = 30

N_COEFF = 100                   # coefficients of U~
NODES = cauchy_nodes()          # Cauchy circle in eps of the reference values: r = 0.2, N = 64
ORDERS = (-1, 0, 1)


# --- the system ---------------------------------------------------------------------------------------------------

_A_NUM = sp.lambdify((X, EPS), connection_matrix(), "mpmath")


def A_boni(x, eps):
    """Boni (A.2), x = t/s, as an mpmath matrix."""
    return mp.matrix(_A_NUM(x, eps).tolist())


_RESIDUE = {}


def balanced_residue_m1(eps):
    """Residue at x = -1 of the balanced matrix A~ = T^-1 A T - T^-1 T', T = diag(1, 1, -x) (sympy, cached)."""
    if "f" not in _RESIDUE:
        T = sp.diag(1, 1, -X)
        At = sp.simplify(T.inv() * connection_matrix() * T - T.inv() * sp.diff(T, X))
        res = ((1 + X) * At).applyfunc(lambda t: sp.limit(sp.simplify(t), X, -1))
        _RESIDUE["f"] = sp.lambdify(EPS, res, "mpmath")
    return mp.matrix(_RESIDUE["f"](eps).tolist())


# --- 1. recursion for U~ ------------------------------------------------------------------------------------------

def _sylvester_solve(L, j, rhs):
    """Solve X (L + j) - L X = rhs for X (3x3), as a linear system for the entries of X."""
    n = L.rows
    K = mp.zeros(n * n, n * n)
    for p in range(n):
        for q in range(n):
            r = p * n + q
            for c in range(n):
                K[r, p * n + c] += L[c, q] + (j if c == q else 0)
                K[r, c * n + q] -= L[p, c]
    v = mp.lu_solve(K, mp.matrix([rhs[p, q] for p in range(n) for q in range(n)]))
    return mp.matrix([[v[p * n + q] for q in range(n)] for p in range(n)])


def frobenius_series(eps, n=N_COEFF):
    """Lambda and [U_0, ..., U_{n-1}] for x A~(x) = Lambda + x A~_-1/(1+x) = Lambda + sum_k (-1)^(k-1) A~_-1 x^k."""
    b0 = 2 * (1 - 2 * eps)
    Lam = mp.matrix([[0, 0, 0], [0, eps, 0], [-b0, 0, 0]])
    Am1 = balanced_residue_m1(eps)
    U = [mp.eye(3)]
    for j in range(1, n):
        rhs = mp.zeros(3, 3)
        for k in range(1, j + 1):
            rhs += (-1) ** (k - 1) * Am1 * U[j - k]
        U.append(_sylvester_solve(Lam, j, rhs))
    return Lam, U


def U_and_dU(U, x):
    """U~(x) and dU~/dx from the coefficients."""
    out, d = mp.zeros(3, 3), mp.zeros(3, 3)
    xp = mp.mpc(1)
    for j, Uj in enumerate(U):
        out += Uj * xp
        if j + 1 < len(U):
            d += (j + 1) * U[j + 1] * xp
        xp *= x
    return out, d


def x_pow(Lam, log_x):
    """x^Lambda = exp(Lambda log x) = [[1,0,0],[0,x^eps,0],[-beta0 log x,0,1]] (Lambda has a Jordan block)."""
    eps, b0 = Lam[1, 1], -Lam[2, 0]
    return mp.matrix([[1, 0, 0], [0, mp.exp(eps * log_x), 0], [-b0 * log_x, 0, 1]])


def H_at(U, x):
    """H(x) = diag(1, 1, -x) U~(x)."""
    return mp.diag([1, 1, -x]) * U_and_dU(U, x)[0]


# --- 2. constants -------------------------------------------------------------------------------------------------

def fix_constants(Lam, U, x0, eps):
    """d from H(x0) x0^Lambda d = M(x0), with the exact M (x0 < 0, ln x0 = ln(-x0) + i pi)."""
    x0 = mp.mpf(x0)
    return mp.lu_solve(H_at(U, x0) * x_pow(Lam, mp.log(-x0) + 1j * mp.pi), exact_M(x0, eps))


# --- 3. the local form --------------------------------------------------------------------------------------------

def local_m3(Lam, U, d, x, analytic=False):
    """Third row of H(x) x^Lambda d and its derivative, at real x < 0.

    analytic=False: ln x = ln(-x) + i pi, the whole function; derivative H' x^Lambda d + H (Lambda/x) x^Lambda d.
    analytic=True:  ln x = i pi, the analytic part;            derivative H' x^Lambda d.
    """
    x = mp.mpf(x)
    Ux, dUx = U_and_dU(U, x)
    H = mp.diag([1, 1, -x]) * Ux
    dH = mp.diag([0, 0, -1]) * Ux + mp.diag([1, 1, -x]) * dUx
    if analytic:
        X = x_pow(Lam, 1j * mp.pi)
        return (H * X * d)[2], (dH * X * d)[2]
    X = x_pow(Lam, mp.log(-x) + 1j * mp.pi)
    return (H * X * d)[2], (dH * X * d + H * Lam * X * d / x)[2]


# --- 4. the exact box ---------------------------------------------------------------------------------------------

def exact_M(x, eps):
    """(M_1, M_2, M_3) from the closed form (BDK, in Boni's basis), real x."""
    return mp.matrix(list(boni_basis(mp.mpf(x), eps)))


def exact_m3(x, eps):
    """Exact M_3 and dM_3/dx, the third row of A(x) M(x)."""
    M = exact_M(x, eps)
    return M[2], (A_boni(x, eps) * M)[2]


# --- the figure data ----------------------------------------------------------------------------------------------

def parts(xs, x0=-0.3, n=N_COEFF, orders=ORDERS):
    """For each x: (x, true, analytic, singular, true', analytic', singular'), each a list over `orders`.

    The singular part is the whole local form minus its analytic part; the constants are fixed at x0."""
    series = [frobenius_series(e, n) for e in NODES]
    ds = [fix_constants(Lam, U, x0, e) for (Lam, U), e in zip(series, NODES)]
    rows = []
    for x in xs:
        tv, td, av, ad, fv, fd = [], [], [], [], [], []
        for (Lam, U), d, e in zip(series, ds, NODES):
            f, fp = local_m3(Lam, U, d, x)
            a, ap = local_m3(Lam, U, d, x, analytic=True)
            t, tp = exact_m3(x, e)
            fv.append(f); fd.append(fp); av.append(a); ad.append(ap); tv.append(t); td.append(tp)
        T, Td, A, Ad, F, Fd = (cauchy_coefficients(v, NODES, orders) for v in (tv, td, av, ad, fv, fd))
        rows.append((float(x), T, A, [f - a for f, a in zip(F, A)], Td, Ad, [f - a for f, a in zip(Fd, Ad)]))
    return rows
