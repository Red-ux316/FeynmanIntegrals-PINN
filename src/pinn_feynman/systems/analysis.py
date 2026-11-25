"""Local analysis of a system dM/dx = A(x, eps) M at its singular points (Frobenius theory).

For each singular point x0 (finite, or infinity via t = 1/x):
  * pole order p of A (Fuchsian point: p <= 1, Henn eq. 4.5-4.6);
  * residue matrix A_0 = lim (x - x0) A  (for x0 = infinity: residue of -A(1/t)/t^2 at t = 0);
  * exponents = eigenvalues of A_0 (the powers (x - x0)^rho of the solutions, Henn eq. 4.9);
  * resonances: pairs of exponents differing by a nonzero integer;
  * Jordan blocks of A_0 (sources of explicit logarithms).

    python -m pinn_feynman.systems.analysis box_massless
"""

import argparse
import importlib
import pkgutil
from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp


class _SystemModules(Mapping):
    """{name: module} of the one-variable systems of this package: every module with `X`, `EPS` and
    `connection_matrix` (found by looking, nothing is listed by hand: a new one-variable system is a new file)."""

    @staticmethod
    def _usable(module):
        return all(hasattr(module, a) for a in ("X", "EPS", "connection_matrix"))

    def _names(self):
        found = []
        for info in pkgutil.iter_modules(importlib.import_module(__package__).__path__):
            if info.name == "analysis":
                continue
            try:
                module = importlib.import_module(f"{__package__}.{info.name}")
            except ImportError:                        # a system still being written must not break the others
                continue
            if self._usable(module):
                found.append(info.name)
        return sorted(found)

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())

    def __getitem__(self, name):
        try:
            module = importlib.import_module(f"{__package__}.{name}")
        except (ImportError, ValueError, TypeError):
            raise KeyError(name) from None
        if name == "analysis" or not self._usable(module):
            raise KeyError(name)
        return module


#: name -> module with connection_matrix(), X, EPS, SINGULAR_POINTS
SYSTEM_MODULES = _SystemModules()
INFINITY = "oo"


@dataclass
class PointAnalysis:
    point: object
    pole_order: int
    residue: sp.Matrix
    exponents: dict            # eigenvalue -> algebraic multiplicity
    resonances: list           # [(rho_i, rho_j, rho_i - rho_j)]
    jordan_blocks: list        # sizes of the Jordan blocks of the residue

    @property
    def fuchsian(self):
        return self.pole_order <= 1


def local_matrix(a, x, point):
    """(matrix, variable t) with the singular point moved to t = 0."""
    t = sp.Symbol("t")
    if point == INFINITY:
        return sp.simplify(-a.subs(x, 1 / t) / t**2), t
    return sp.simplify(a.subs(x, point + t)), t


def pole_order(a, t):
    """Highest order of the pole at t = 0 over the entries of a (0 if regular)."""
    order = 0
    for entry in a:
        entry = sp.together(entry)
        if entry == 0:
            continue
        num, den = sp.fraction(sp.cancel(entry))
        if den.has(t):                                   # multiplicity of the root t = 0 of the denominator
            order = max(order, sp.roots(sp.Poly(den, t)).get(0, 0))
    return order


def residue(a, t):
    """lim_{t -> 0} t * a (entrywise)."""
    return (t * a).applyfunc(lambda e: sp.simplify(sp.limit(sp.simplify(e), t, 0)))


def resonances(exponents):
    """Pairs of exponents whose difference is a nonzero integer (for generic eps)."""
    out = []
    eigs = list(exponents)
    for i, a in enumerate(eigs):
        for b in eigs[i + 1:]:
            diff = sp.simplify(a - b)
            if diff.is_integer and diff != 0:
                out.append((a, b, diff))
    return out


def jordan_blocks(m):
    """Sizes of the Jordan blocks of m (generic eps)."""
    _, j = m.jordan_form()
    sizes, size = [], 1
    for k in range(j.rows - 1):
        if j[k, k + 1] != 0:
            size += 1
        else:
            sizes.append(size)
            size = 1
    sizes.append(size)
    return sizes


def analyse_point(a, x, point):
    local, t = local_matrix(a, x, point)
    order = pole_order(local, t)
    res = residue(local, t) if order <= 1 else None
    exps = res.eigenvals() if res is not None else {}
    return PointAnalysis(point=point, pole_order=order, residue=res, exponents=exps,
                         resonances=resonances(exps), jordan_blocks=jordan_blocks(res) if res is not None else [])


def analyse(name, include_infinity=True):
    """PointAnalysis for every singular point of a registered system."""
    module = SYSTEM_MODULES[name]
    a = module.connection_matrix()
    points = list(module.SINGULAR_POINTS) + ([INFINITY] if include_infinity else [])
    return [analyse_point(a, module.X, p) for p in points]


def report(analyses):
    lines = ["| point | pole order | Fuchsian | residue | exponents (multiplicity) | resonances | Jordan blocks |",
             "|---|---|---|---|---|---|---|"]
    for r in analyses:
        if not r.fuchsian:
            lines.append(f"| {r.point} | {r.pole_order} | no (pole of order {r.pole_order} in this basis) | — | — | — | — |")
            continue
        exps = ", ".join(f"{sp.sstr(k)} ({v})" for k, v in r.exponents.items())
        res = ", ".join(f"{sp.sstr(a)}-{sp.sstr(b)}={sp.sstr(d)}" for a, b, d in r.resonances) or "none"
        lines.append(f"| {r.point} | {r.pole_order} | yes | `{sp.sstr(r.residue.tolist())}` | "
                     f"{exps} | {res} | {r.jordan_blocks} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=sorted(SYSTEM_MODULES))
    args = parser.parse_args(argv)
    text = report(analyse(args.system))
    print(text)
    return text


if __name__ == "__main__":
    main()
