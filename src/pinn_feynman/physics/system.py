"""Connection matrices A = sum_j eps^j A_j as torch tensors, in the network's input variables.

A system has d variables v_1..v_d (`pinn_feynman.systems`: the massless box has d = 1, x = t/s; the one-mass box
d = 2, (s1, s2) at s0 = 1) and one matrix per variable, dM/dv_k = A_k(v) M. The network may use rescaled variables
z = c v (Boni: x_B = 10 t/s; one-mass: z = 10 (s1, s2)); by the chain rule
    dM/dz_k = (1/c) A_k(z/c) M.

Shapes (the direction axis only with more than one variable):
- d = 1: z of shape (N, 1) or (N,)  ->  A0, A1 of shape (N, M, M);
- d > 1: z of shape (N, d)          ->  A0, A1 of shape (N, d, M, M), axis 1 = direction k.

The systems of the full basis are linear in eps (`matrices`); `coefficient_orders` and `coefficients` also serve a
higher degree in eps and rectangular blocks (the A_UK of the bottom-up formulation, `BottomUpSystem` in
`pinn_feynman.physics.bottom_up`). `coefficients(z)` is what the residual consumers use.
"""

import sympy as sp
import torch

from ..problems import ProblemMapping, available_problems, get_problem
from .coefficients import ResidualCoefficients

#: problem name -> module with VARIABLES, connection_matrix_orders(variable), N_MASTERS (read from `pinn_feynman.problems`)
SYSTEMS = ProblemMapping(lambda problem: problem.system_module())


def _entry_function(entry, symbols):
    """Callable (v_1, ..., v_d) -> entry(v) that works on torch tensors (rational entries only)."""
    if entry.free_symbols - set(symbols):
        raise ValueError(f"matrix entry {entry} depends on symbols other than {tuple(symbols)}")
    if entry.is_number:
        value = float(entry)
        return lambda *v: torch.full_like(v[0], value)
    return sp.lambdify(symbols[0] if len(symbols) == 1 else tuple(symbols), entry, modules="math")


def _evaluate(entries, *v):
    return torch.stack([torch.stack([f(*v) for f in row], dim=-1) for row in entries], dim=-2)


def _entries(matrix, symbols):
    return [[_entry_function(e, symbols) for e in matrix.row(i)] for i in range(matrix.rows)]


class TorchSystem:
    """A0(z), A1(z) per variable, as tensors (shapes in the module docstring).

    `TorchSystem(a0, a1, symbol, input_scale)` builds a one-variable system;
    `TorchSystem.from_orders` and `TorchSystem.from_module` build one with any number of variables.
    """

    def __init__(self, a0, a1, symbol, input_scale=1.0):
        self._setup([(a0, a1)], (symbol,), (str(symbol),), input_scale)

    @classmethod
    def from_orders(cls, orders, symbols, input_scale=1.0, variables=None):
        """System from [(A0_k, A1_k) for each variable k] (sympy Matrices in `symbols`, one symbol per variable)."""
        symbols = tuple(symbols)
        if len(orders) != len(symbols):
            raise ValueError(f"{len(orders)} pairs of matrices for {len(symbols)} variables")
        system = cls.__new__(cls)
        system._setup(orders, symbols, tuple(variables) if variables is not None else tuple(map(str, symbols)),
                      input_scale)
        return system

    @classmethod
    def from_module(cls, module, input_scale=1.0):
        """System of a module of `pinn_feynman.systems` (VARIABLES, connection_matrix_orders(variable))."""
        variables = tuple(module.VARIABLES)
        orders = [module.connection_matrix_orders(v) for v in variables]
        return cls.from_orders(orders, [sp.Symbol(v) for v in variables], input_scale, variables)

    def _setup(self, orders, symbols, variables, input_scale):
        # orders: per variable a tuple (A_0, ..., A_p) of sympy Matrices (rectangular allowed); shorter tuples are
        # completed with zero matrices up to the common degree p >= 1
        degree = max(1, max(len(o) for o in orders) - 1)
        shape = orders[0][0].shape
        orders = [tuple(o) + (sp.zeros(*shape),) * (degree + 1 - len(o)) for o in orders]
        self.n_masters, self.n_columns = shape
        self.degree = degree
        self.cheap = True
        self.variables = variables
        self.n_variables = len(variables)
        self.input_scale = float(input_scale)
        self._orders = [[_entries(o[j], symbols) for o in orders] for j in range(degree + 1)]
        self._a0, self._a1 = self._orders[0], self._orders[1]

    def matrices(self, z):
        """(A0, A1) at the points z in the network variables: (N, M, M) for d = 1, (N, d, M, M) for d > 1."""
        if self.degree != 1:
            raise ValueError(f"the system has degree {self.degree} in eps: use coefficients(z)")
        if self.n_variables == 1:
            # one variable: no direction axis
            x = z.reshape(-1) / self.input_scale
            return _evaluate(self._a0[0], x) / self.input_scale, _evaluate(self._a1[0], x) / self.input_scale
        if z.dim() != 2 or z.shape[1] != self.n_variables:
            raise ValueError(f"points of a system in {self.n_variables} variables {self.variables} must have shape "
                             f"(N, {self.n_variables}), got {tuple(z.shape)}")
        v = [z[:, k] / self.input_scale for k in range(self.n_variables)]
        a0 = torch.stack([_evaluate(m, *v) for m in self._a0], dim=1)
        a1 = torch.stack([_evaluate(m, *v) for m in self._a1], dim=1)
        return a0 / self.input_scale, a1 / self.input_scale

    def coefficient_orders(self, z):
        """(A_0, ..., A_p) at the points z, shapes as `matrices` (rectangular when the block is)."""
        if self.degree == 1:
            return self.matrices(z)
        if self.n_variables == 1:
            x = z.reshape(-1) / self.input_scale
            return tuple(_evaluate(entries[0], x) / self.input_scale for entries in self._orders)
        if z.dim() != 2 or z.shape[1] != self.n_variables:
            raise ValueError(f"points of a system in {self.n_variables} variables {self.variables} must have shape "
                             f"(N, {self.n_variables}), got {tuple(z.shape)}")
        v = [z[:, k] / self.input_scale for k in range(self.n_variables)]
        return tuple(torch.stack([_evaluate(m, *v) for m in entries], dim=1) / self.input_scale
                     for entries in self._orders)

    def coefficients(self, z):
        """The coefficients of the residual at z: A_0..A_p and no forcing (the complete system)."""
        return ResidualCoefficients(tuple(self.coefficient_orders(z)))


def _module(name):
    try:
        return SYSTEMS[name]
    except KeyError:
        raise ValueError(f"unknown system {name!r}; available: {available_problems()}") from None


def system_variables(name):
    """Names of the variables of a registered problem (without building its matrices)."""
    return tuple(_module(name).VARIABLES)


def build_system(name, input_scale=1.0, part=None):
    """System of a registered problem, in the variables z = input_scale * v.

    A problem in the full basis gives a `TorchSystem` (`part` is ignored). A bottom-up problem (it declares
    `unknowns` and a `provider`) gives a `BottomUpSystem` for the `part` ('real' or 'imag') asked.
    """
    problem = get_problem(name) if name in available_problems() else None
    if problem is None or problem.provider is None:
        return TorchSystem.from_module(_module(name), input_scale=input_scale)
    if part is None:
        raise ValueError(f"{name!r} is a bottom-up problem: build_system needs part='real' or 'imag'")
    from ..data.generation import generator_info
    from ..systems.blocks import split
    from .bottom_up import BottomUpSystem

    module = problem.system_module()
    variables = tuple(module.VARIABLES)
    symbols = [sp.Symbol(v) for v in variables]
    parts = split([tuple(module.connection_matrix_orders(v)) for v in variables], problem.unknowns)
    uu = TorchSystem.from_orders(parts.uu, symbols, input_scale, variables)
    uk = TorchSystem.from_orders(parts.uk, symbols, input_scale, variables) if parts.uk is not None else None
    orders = tuple(generator_info(problem.data_problem())[1]["orders"])
    provider = problem.build_provider()
    return BottomUpSystem(uu, uk, provider, part, orders, unknowns=parts.unknowns, knowns=parts.knowns)
