"""Dashed vertical lines at the singular points of a problem, on every figure with the domain on the x axis.

One place for the singular points of each problem: the positions come from the system modules (`SINGULAR_POINTS` of
the matrix and, when declared, `FORCING_BRANCH_POINTS` of the known term; for several variables the
`DENOMINATOR_FACTORS`); this module only adds a short label and the kind of each point:
- ``physical``: a pole of the matrix of the TOTAL system (not of the bottom-up block) where at least one master
  integral is singular (dark red);
- ``spurious``: a pole of that matrix where all the master integrals are analytic (grey-blue).

The kinds are those of the documents of the true values (the same as the scripts of their figures), kept in
`systems/singular_kinds.py`; a point of a system module without a kind there is an error.

Use: ``mark(ax, singular_points(problem))`` after the data are drawn, so the axis limits are final. Points outside the
x range are skipped (the axis limits never change); on a logarithmic axis the points <= 0 are skipped. For several
variables: ``mark_lines(ax, singular_lines(problem))`` draws the singular lines a*v1 + b*v2 = c on a map, clipped to it.
"""

import importlib

import numpy as np

PHYSICAL = "#a11d2e"
SPURIOUS = "#4a6fa5"
DASHES = (0, (5, 3))

#: kinds of the points and lines: one registry, in systems/singular_kinds.py (also used by the distance weighting)
from ..systems.singular_kinds import KINDS, LINE_KINDS  # noqa: E402


#: problem -> module of `systems/` when it has another name
SYSTEM_MODULES = {"sunrise_equal_mass_positive": "sunrise_equal_mass"}


def _system(problem):
    return importlib.import_module(f"pinn_feynman.systems.{SYSTEM_MODULES.get(problem, problem)}")


def positions(problem):
    """The singular points of a one-variable problem read from its system module (matrix and known term), sorted."""
    module = _system(problem)
    points = set(getattr(module, "SINGULAR_POINTS", ())) | set(getattr(module, "FORCING_BRANCH_POINTS", ()))
    return sorted(float(p) for p in points)


def singular_points(problem, scale=1.0, absolute=False, sign=-1.0, name=None):
    """[(position on the axis, label, kind)] of a one-variable problem; [] for a problem in several variables.

    `scale`: the axis shows scale * x (e.g. Boni's x_B = 10 x of the massless box); `absolute`: the axis shows |x| and
    only the points on the side `sign` of x = 0 (and x = 0) are kept; `name`: the variable in the labels (default the
    first of the system's VARIABLES). Labels keep the sign of the point, e.g. "x=-4 (threshold)"."""
    module = _system(problem)
    if len(getattr(module, "VARIABLES", ("x",))) != 1:
        return []
    kinds = KINDS.get(problem, {})
    name = name or module.VARIABLES[0]
    out = []
    for p in positions(problem):
        if p not in kinds:
            raise KeyError(f"{problem}: singular point {p:g} has no kind in systems/singular_kinds.py")
        note, kind = kinds[p]
        value = scale * p
        if absolute:
            if p * sign < 0:
                continue
            value = abs(value)
        label = f"{name}={scale * p:g}" + (f" ({note})" if note else "")
        out.append((float(value), label, kind))
    return out


def singular_lines(problem, s0=1):
    """[(a, b, c, label, kind)]: the singular lines a*v1 + b*v2 = c of a two-variable problem, from the
    DENOMINATOR_FACTORS of its system module at s0 = 1 (constant factors dropped); [] for one variable."""
    import sympy as sp

    module = _system(problem)
    variables = getattr(module, "VARIABLES", ())
    if len(variables) != 2 or not hasattr(module, "DENOMINATOR_FACTORS"):
        return []
    kinds = LINE_KINDS.get(problem, {})
    v1, v2 = (sp.Symbol(v) for v in variables)
    s0_symbol = getattr(module, "S0", None)
    out = []
    for factor in module.DENOMINATOR_FACTORS:
        expr = sp.expand(factor.subs(s0_symbol, s0) if s0_symbol is not None else factor)
        if not expr.free_symbols:
            continue
        key = str(expr)
        if key not in kinds:
            raise KeyError(f"{problem}: singular line {key} = 0 has no kind in systems/singular_kinds.py")
        a, b = float(expr.coeff(v1)), float(expr.coeff(v2))
        c = -float(expr.subs({v1: 0, v2: 0})) + 0.0
        out.append((a, b, c, *kinds[key]))
    return out


def colour(kind):
    return PHYSICAL if kind == "physical" else SPURIOUS


def mark(ax, points, label_y=0.97):
    """Dashed vertical line and a small rotated label at each (x, label, kind) inside the x range of `ax`; the axis
    limits do not change. Returns the number of lines drawn."""
    if not points:
        return 0
    lo, hi = ax.get_xlim()
    lo, hi = min(lo, hi), max(lo, hi)
    log = ax.get_xscale() == "log"
    drawn = 0
    for x, label, kind in points:
        if not lo <= x <= hi or (log and x <= 0):
            continue
        ax.axvline(x, color=colour(kind), ls=DASHES, lw=1.1, alpha=0.9, zorder=1)
        ax.text(x, label_y, label, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top",
                fontsize=6.5, color=colour(kind))
        drawn += 1
    return drawn


def mark_lines(ax, lines):
    """Dashed singular lines a*v1 + b*v2 = c on a map, clipped to the current limits (unchanged), each with a label
    near its middle. Returns the number of lines that cross the visible area."""
    if not lines:
        return 0
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    drawn = 0
    for a, b, c, label, kind in lines:
        if abs(b) > abs(a):
            xs = np.linspace(x0, x1, 200)
            ys = (c - a * xs) / b
        else:
            ys = np.linspace(y0, y1, 200)
            xs = (c - b * ys) / a
        inside = (xs >= min(x0, x1)) & (xs <= max(x0, x1)) & (ys >= min(y0, y1)) & (ys <= max(y0, y1))
        if not inside.any():
            continue
        ax.plot(np.where(inside, xs, np.nan), np.where(inside, ys, np.nan), color=colour(kind), ls=DASHES, lw=1.0,
                alpha=0.9, zorder=3)
        middle = np.flatnonzero(inside)[inside.sum() // 2]
        ax.text(xs[middle], ys[middle], label, fontsize=6, color=colour(kind), ha="left", va="bottom")
        drawn += 1
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return drawn
