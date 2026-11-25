"""Distance weighting (DW): physics residual multiplied by the distance from the nearest pole of the block A_UU.

For the error e of the network the residual is r = e' - A_UU e (the known term of the lower sectors cancels): near a
pole x_0 of A_UU, A_UU ~ A_0 / (x - x_0), so the same error weighs like 1 / |x - x_0| and the plain mean squared
residual is dominated by the points next to the pole. The weight w(x) = d(x) / c, with d the distance from the
nearest pole, turns the equation into its Euler form, d r of order e everywhere: a preconditioning of the operator,
independent of how the solution grows (the loss is <(w r)^2>, `weighted_physics_loss`).

The poles are **computed** from the system, not read from a list written by hand:

- A_UU is the block of `connection_matrix_orders(variable)` on the columns learnt by the network
  (`Problem.columns()`; all of them for a problem in the full basis), every order in eps.
- **Physical and spurious** poles count alike; the branch points of the known term only (x = -4
  of the non-planar triangle) do not, since A_UU is regular there. The poles of the lower block and of the mixed
  block A_UK do not count either: they never multiply the error of the network.
- One variable: d(x) = min over the real roots of the denominators of A_UU.
- Two variables (one-mass box): every denominator factor must be linear, a v1 + b v2 = c; d = min over the lines of
  the Euclidean distance |a v1 + b v2 - c| / sqrt(a^2 + b^2), at s0 = `S0_VALUE`.
- Distances are measured in the physical variables v = z / input_scale, not in the network variables z.
- c = RMS of d over a fixed uniform grid of the training domain, computed once: the weights have mean square ~1 on
  the domain, whatever the sampler, so the scale of L_PDE stays comparable with the unweighted one.

The weight depends only on the position: no state, valid with fixed base points and with fresh points (resample, R3).
"""

from functools import lru_cache

import numpy as np
import sympy as sp
import torch

#: points of the fixed grid used for the normalisation constant
NORMALISATION_POINTS = 20_000


def _problem(system):
    """(system module, columns learnt by the network or None) of the problem named `system`."""
    from ..problems import get_problem

    problem = get_problem(system)
    return problem.system_module(), problem.columns()


def _variables(module):
    return tuple(getattr(module, "VARIABLES", ("x",)))


def _symbol(expr, name):
    """The sympy symbol called `name` among the free symbols of `expr` (the modules create their own)."""
    return next((s for s in expr.free_symbols if s.name == name), sp.Symbol(name))


def _block_factors(module, columns):
    """The distinct non-constant irreducible factors of the denominators of A_UU, over every variable and order."""
    factors = set()
    for variable in _variables(module):
        for order in module.connection_matrix_orders(variable):
            index = list(columns) if columns is not None else list(range(order.rows))
            for entry in order.extract(index, index):
                if entry == 0:
                    continue
                _, denominator = sp.fraction(sp.factor(sp.together(entry)))
                for factor, _ in sp.factor_list(denominator)[1]:
                    if factor.free_symbols:
                        factors.add(sp.expand(factor))
    return factors


@lru_cache(maxsize=None)
def block_poles(system):
    """Real poles of A_UU of a one-variable problem (float, sorted)."""
    module, columns = _problem(system)
    (name,) = _variables(module)
    poles = set()
    for factor in _block_factors(module, columns):
        poly = sp.Poly(factor, _symbol(factor, name))
        poles |= {float(r) for r in sp.real_roots(poly)}
    if not poles:
        raise ValueError(f"{system}: A_UU has no real pole, the distance weight has nothing to measure from")
    return tuple(sorted(poles))


@lru_cache(maxsize=None)
def block_lines(system):
    """((a, b, c), ...) of the singular lines a v1 + b v2 = c of A_UU of a two-variable problem, with a^2 + b^2 = 1
    (sorted)."""
    module, columns = _problem(system)
    names = _variables(module)
    out = set()
    for factor in _block_factors(module, columns):
        v1, v2 = (_symbol(factor, n) for n in names)
        if factor.free_symbols - {v1, v2}:
            raise ValueError(f"{system}: denominator factor {factor} of A_UU has symbols other than {names}")
        poly = sp.Poly(factor, v1, v2)
        if poly.total_degree() != 1:
            raise ValueError(f"{system}: denominator factor {factor} of A_UU is not a line in {names}")
        a, b = float(poly.coeff_monomial(v1)), float(poly.coeff_monomial(v2))
        c = -float(poly.coeff_monomial(1)) + 0.0
        norm = float(np.hypot(a, b))
        out.add((a / norm, b / norm, c / norm))
    if not out:
        raise ValueError(f"{system}: A_UU has no singular line, the distance weight has nothing to measure from")
    return tuple(sorted(out))


def distance_function(system):
    """d(v) for points v (N, d) in the physical variables (numpy or torch, float64): distance from the nearest pole
    (one variable) or singular line (two variables) of A_UU."""
    module, _ = _problem(system)
    n_variables = len(_variables(module))
    if n_variables == 1:
        points = block_poles(system)

        def distance(v):
            x = v.reshape(-1)
            if isinstance(x, torch.Tensor):
                return torch.stack([torch.abs(x - p) for p in points]).min(dim=0).values
            return np.min([np.abs(x - p) for p in points], axis=0)

        return distance
    if n_variables == 2:
        lines = block_lines(system)

        def distance(v):
            v = v.reshape(-1, 2)
            terms = [abs(a * v[:, 0] + b * v[:, 1] - c) for a, b, c in lines]
            if isinstance(v, torch.Tensor):
                return torch.stack(terms).min(dim=0).values
            return np.min(terms, axis=0)

        return distance
    raise ValueError(f"distance weight: systems of 1 or 2 variables, got {n_variables}")


def _grid(domain, n):
    """A fixed uniform grid of the domain (network variables): linspace for an interval, lattice for the triangle."""
    if getattr(domain, "dim", 1) == 2:
        return np.asarray(domain.grid(n), dtype=float)
    lo, hi = (float(v) for v in domain)
    return np.linspace(lo, hi, n).reshape(-1, 1)


class DistanceWeighting:
    """w = d(z / input_scale) / c: DW, the residual times the distance from the nearest pole or singular line of
    A_UU."""

    per_point_state = False              # a function of the position only
    multi_variable = True                # one variable (points) and two variables (lines)
    log_unweighted = True                # the trainer logs the plain L_PDE too (`physics_loss_unweighted`)

    @classmethod
    def build(cls, params, n_base=None, domain=None, system=None, input_scale=1.0):
        if system is None:
            raise ValueError("point weighting 'distance' needs the name of the system")
        return cls(system, domain, input_scale=input_scale, **params)

    def __init__(self, system, domain, input_scale=1.0):
        self.system = system
        self.input_scale = float(input_scale)
        self.distance = distance_function(system)
        d = self.distance(_grid(domain, NORMALISATION_POINTS) / self.input_scale)
        if np.any(d <= 0):
            raise ValueError(f"distance weight: the domain {domain!r} touches a pole of A_UU of {system!r}")
        self.normalisation = float(np.sqrt(np.mean(d**2)))

    def weights(self, z):
        """w (N,) at the points z (N, d) of the network variables, in the dtype of z."""
        v = z.detach().to(torch.float64) / self.input_scale
        return (self.distance(v) / self.normalisation).to(dtype=z.dtype, device=z.device)

    def __call__(self, idx, z, residual_norm):
        return self.weights(z)

    def end_epoch(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
