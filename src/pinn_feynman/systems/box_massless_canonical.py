"""One-loop massless box in a canonical (epsilon-factorised, dlog) basis.

Basis of CMZ (arXiv:2312.02067) eq. (2.9): F~ = (s t I_1111, s I_1110, t I_1101), obtained from
Boni's basis M by F~ = T M:

    T = [[0, 0, s t^(-1-eps)],
         [0, (2 eps - 1)/eps t^(-eps), 0],
         [(2 eps - 1)/eps t^(-eps), 0, 0]],        t = x s.

In the variable x = t/s:
    dF~/dx = eps [ a/x + b/(1+x) ] F~,   a = [[-1,0,-2],[0,0,0],[0,0,-1]],  b = [[1,2,2],[0,0,0],[0,0,0]].

Used for comparing the local (Frobenius) structure with Boni's basis (systems/analysis.py):
the resonance of Boni's basis at x = 0 becomes a Jordan block, and x = infinity is Fuchsian.
"""

import sympy as sp

from .box_massless import EPS, X

SINGULAR_POINTS = (0, -1)
N_MASTERS = 3
S = sp.Symbol("s", positive=True)

A_RES = sp.Matrix([[-1, 0, -2], [0, 0, 0], [0, 0, -1]])
B_RES = sp.Matrix([[1, 2, 2], [0, 0, 0], [0, 0, 0]])


def connection_matrix():
    """eps [a/x + b/(1+x)]."""
    return EPS * (A_RES / X + B_RES / (1 + X))


def gauge_from_boni():
    """T(x, s, eps) with F~ = T M (M = Boni's basis), t = x s."""
    t = X * S
    c = (2 * EPS - 1) / EPS
    return sp.Matrix([[0, 0, S * t ** (-1 - EPS)], [0, c * t ** (-EPS), 0], [c * t ** (-EPS), 0, 0]])
