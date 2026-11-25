"""What to generate (DatasetSpec) and what is stored (ReferenceDataset)."""

from dataclasses import dataclass, fields

import numpy as np
import torch

from .domains import Interval, Triangle, as_domain

#: real and imaginary parts are Boni's two separate trainings (sec. 6.2.1.1)
PARTS = ("real", "imag")
#: how reference values are obtained: exact evaluation at every point, or exact values at
#: Chebyshev nodes and interpolation (for large test sets; one variable only)
METHODS = ("exact", "chebyshev")


@dataclass(frozen=True)
class DatasetSpec:
    """A set of points and the problem whose reference values are computed on them.

    `domain` accepts any configuration form of a domain (pair [lo, hi], dict {name: triangle,
    ...}, or a domain object) and is stored as a domain object (`data.domains`); an interval
    is a tuple (lo, hi), so its cache key is that of the pair (lo, hi).
    """

    problem: str
    sampler: str
    domain: Interval | Triangle
    n: int
    seed: int | None = None
    method: str = "exact"               # "exact" or "chebyshev"
    n_nodes: int | None = None          # Chebyshev nodes, for method "chebyshev"

    def __post_init__(self):
        object.__setattr__(self, "domain", as_domain(self.domain))
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.method == "chebyshev" and not self.n_nodes:
            raise ValueError("method 'chebyshev' needs n_nodes")
        if self.domain.dim > 1 and "chebyshev" in (self.method, self.sampler):
            raise ValueError(f"Chebyshev nodes and interpolation are one-dimensional: not available on the "
                             f"{self.domain.dim}-dimensional domain {self.domain.label()} (use method 'exact')")

    @classmethod
    def from_config(cls, problem, domain, entry):
        """Build from a `data.datasets.<name>` entry of the YAML configuration."""
        return cls(problem=problem, sampler=entry["sampler"], domain=as_domain(domain),
                   n=int(entry.get("n", 2)), seed=entry.get("seed"), method=entry.get("method", "exact"),
                   n_nodes=entry.get("n_nodes"))

    def as_dict(self):
        """JSON form (it enters the cache key): the domain in its configuration form."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["domain"] = self.domain.as_config()
        return d


def select_part(values, part):
    """Real or imaginary part of a complex array."""
    if part == "real":
        return values.real.copy()
    if part == "imag":
        return values.imag.copy()
    raise ValueError(f"part must be one of {PARTS}, got {part!r}")


@dataclass
class ReferenceDataset:
    """Points x and complex reference values (N, N_masters, N_orders).

    x has shape (N, d) on a d-dimensional domain, and (N,) on an interval (one
    variable, no trailing axis).
    """

    spec: DatasetSpec
    x: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        if self.values.shape[0] != self.x.shape[0]:
            raise ValueError("x and values have different numbers of points")
        dim = self.spec.domain.dim
        if dim > 1 and (self.x.ndim != 2 or self.x.shape[1] != dim):
            raise ValueError(f"points of a {dim}-dimensional domain must have shape (N, {dim}), got {self.x.shape}")

    def __len__(self):
        return len(self.x)

    def part(self, part):
        """Real array (N, N_masters, N_orders): the targets of one training."""
        return select_part(self.values, part)

    def tensors(self, part, dtype=torch.float64):
        """(x of shape (N, d), targets of shape (N, N_masters, N_orders)) as torch tensors (d = 1: (N, 1))."""
        x = torch.as_tensor(self.x, dtype=dtype).reshape(-1, self.spec.domain.dim)
        y = torch.as_tensor(self.part(part), dtype=dtype)
        return x, y
