"""The registry of problems: finds a problem by name, declares none.

A problem (a Feynman diagram with its system of differential equations and its true values) is declared in its own
file `problems/<name>.py`, which defines one constant `PROBLEM = Problem(name="<name>", ...)`. This module only knows
how to find that file (`get_problem`, `available_problems`) and how to turn the references it holds into modules
(`Problem.system_module`, `Problem.reference_module`). Nothing is imported until it is asked for: a problem
costs one tiny import, sympy and mpmath come in only when its system or its reference is used (the Boni results import
`pinn_feynman.evaluation`, hence torch).
"""

import importlib
import pkgutil
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache


def _resolve(reference):
    """'package.module' -> the module; 'package.module:NAME' -> that attribute of the module."""
    module, _, attribute = reference.partition(":")
    loaded = importlib.import_module(module)
    return getattr(loaded, attribute) if attribute else loaded


class UnknownProblem(ValueError):
    """No file `problems/<name>.py` for this name (any other ValueError is a file that declares itself badly)."""


@dataclass(frozen=True)
class Problem:
    """What the pipeline needs to know about a diagram. Every reference is a dotted string, resolved on first use.

    name       the name of the file that declares it (the value of `data.problem` and `physics.system` in the configs)
    system     module of `pinn_feynman.systems`: VARIABLES, N_MASTERS, connection_matrix_orders(variable)
    reference  module of `pinn_feynman.reference` that gives the true values: reference_values(points) ->
               complex array (N, N_masters, N_orders), ORDERS, CACHE_PARAMS (what identifies its output, enters the
               cache key), COMPUTE_OPTIONS (how the points are split: never changes the values)
    boni       'module:NAME' of the dictionary of Boni's results for this problem (optional)

    Bottom-up variant: `unknowns` are the columns of the basis learnt by the network, the others are
    supplied by `provider` ('module:factory', a factory without arguments that builds the provider cheaply).
    `base` names the problem whose data, cache, reference and Boni results are used (`system` and `reference`
    must be the same strings as the base's): a bottom-up problem never has data of its own.

    parts      the parts of the true values that can be trained: ('real',) when the imaginary part vanishes
               identically in the domain, so that a training on zeros stops with an error
    """

    name: str
    system: str
    reference: str
    boni: str | None = None
    base: str | None = None
    unknowns: tuple | None = None
    provider: str | None = None
    parts: tuple = ("real", "imag")

    def system_module(self):
        return _resolve(self.system)

    def reference_module(self):
        return _resolve(self.reference)

    def generator(self):
        return self.reference_module().reference_values

    def cache_params(self):
        return self.reference_module().CACHE_PARAMS

    def compute_options(self):
        return self.reference_module().COMPUTE_OPTIONS

    def data_problem(self):
        """Name of the problem whose data are used: the base of a bottom-up problem, else its own name."""
        return self.base or self.name

    def columns(self):
        """Columns of the basis learnt by the network (None = all)."""
        return self.unknowns

    def build_provider(self):
        """The provider of the knowns (None for a problem in the full basis)."""
        return _resolve(self.provider)() if self.provider else None

    def boni_results(self):
        return _resolve(self.boni) if self.boni else {}


def available_problems():
    """Names of the declared problems, sorted: the files of this package that do not start with an underscore."""
    package = importlib.import_module(__package__)
    return sorted(m.name for m in pkgutil.iter_modules(package.__path__) if not m.name.startswith("_"))


@lru_cache(maxsize=None)
def get_problem(name):
    """The `Problem` declared in `problems/<name>.py`; ValueError if there is no such file or it declares it badly."""
    unknown = UnknownProblem(f"unknown problem {name!r}; available: {available_problems()}")
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise unknown
    module_name = f"{__package__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise unknown from None
        raise
    problem = getattr(module, "PROBLEM", None)
    if not isinstance(problem, Problem):
        raise ValueError(f"{module_name} must define PROBLEM = Problem(...)")
    if problem.name != name:
        raise ValueError(f"{module_name} declares the problem {problem.name!r}: the name must be that of the file")
    _check_bottom_up(problem)
    return problem


def _check_bottom_up(problem):
    if not problem.parts or not set(problem.parts) <= {"real", "imag"}:
        raise ValueError(f"{problem.name}: `parts` must be a non-empty subset of ('real', 'imag'), got {problem.parts}")
    if problem.unknowns is not None and problem.provider is None:
        raise ValueError(f"{problem.name}: `unknowns` needs a `provider`")
    if problem.provider is not None and problem.unknowns is None:
        raise ValueError(f"{problem.name}: a `provider` needs `unknowns`")
    if problem.base is None:
        return
    if problem.unknowns is None:
        raise ValueError(f"{problem.name}: `base` is for bottom-up problems (`unknowns`, `provider`)")
    base = get_problem(problem.base)
    if base.base is not None:
        raise ValueError(f"{problem.name}: the base {problem.base!r} is itself a bottom-up problem")
    if (problem.system, problem.reference) != (base.system, base.reference):
        raise ValueError(f"{problem.name}: `system` and `reference` must be those of the base {problem.base!r}")
    if not set(problem.parts) <= set(base.parts):
        raise ValueError(f"{problem.name}: `parts` {problem.parts} must be among those of the base {base.parts}")


class ProblemMapping(Mapping):
    """Read-only view {problem name: value}, built from the registry and resolved only when an item is read.

    Behind `physics.SYSTEMS`, `data.generation.GENERATORS`, ...
    `value(problem)` gives the item; `present(problem)` says whether the problem has one at all.
    """

    def __init__(self, value, present=lambda problem: True):
        self._value, self._present = value, present

    def __iter__(self):
        return iter([n for n in available_problems() if self._present(get_problem(n))])

    def __len__(self):
        return len(list(iter(self)))

    def __getitem__(self, name):
        try:
            problem = get_problem(name)
        except UnknownProblem:
            raise KeyError(name) from None
        if not self._present(problem):
            raise KeyError(name)
        return self._value(problem)

    def __contains__(self, name):
        try:
            self[name]
        except KeyError:
            return False
        return True
