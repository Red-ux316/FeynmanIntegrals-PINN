"""Domains of the kinematic variables: an interval (one variable) or the one-mass triangle (two).

A domain knows its dimension, samples points uniformly, gives its boundary points, tests
membership, rescales to the network variable and round-trips through the configuration:

    data.domain: [-0.9, -0.1]                       ->  Interval(-0.9, -0.1)   (massless box, x = t/s)
    data.domain: {name: triangle, margin: 0.1}      ->  Triangle(margin=0.1)   (one-mass box, (s1, s2))

Points of a d-dimensional domain have shape (N, d), except for the interval, whose points
have shape (N,). The collocation samplers use `sample_columns` (numpy, (N, d) also for the
interval) and `sample_torch` (torch generator, (N, d)).

`Interval` is a tuple (lo, hi): it compares, hashes and unpacks like the pair, and its
configuration form is the list [lo, hi], so the cache key of an interval depends only on its
end points.
"""

from dataclasses import dataclass

import importlib

import numpy as np
import torch


def _one_mass():
    """systems.box_one_mass, imported on first use (it imports sympy: about 1.5 s)."""
    return importlib.import_module("pinn_feynman.systems.box_one_mass")


def _rng(seed):
    """numpy Generator from a seed (int or None) or an existing Generator."""
    return np.random.default_rng(seed)


class Interval(tuple):
    """Closed interval [lo, hi] of one variable; a tuple (lo, hi) of floats."""

    name = "interval"
    dim = 1

    def __new__(cls, lo, hi=None):
        if hi is None:                       # Interval((lo, hi)): also used by copy and pickle
            lo, hi = lo
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"an interval needs finite lo < hi, got ({lo}, {hi})")
        return super().__new__(cls, (lo, hi))

    @property
    def lo(self):
        return self[0]

    @property
    def hi(self):
        return self[1]

    def __repr__(self):
        return f"Interval({self.lo!r}, {self.hi!r})"

    def sample(self, n, seed=None):
        """n uniform points, shape (n,); `seed`: int, None or numpy Generator. Same stream as
        `np.random.default_rng(seed).uniform(lo, hi, size=n)`."""
        return _rng(seed).uniform(self.lo, self.hi, size=n)

    def sample_columns(self, n, seed=None):
        """n uniform points as a column, shape (n, 1): `rng.uniform(lo, hi, size=(n, 1))`, the base points of the
        collocation samplers (Dirichlet probabilities, if any, come next from the same generator)."""
        return _rng(seed).uniform(self.lo, self.hi, size=(n, 1))

    def sample_torch(self, n, generator, dtype=torch.float64):
        """n uniform points (n, 1) from a torch generator: `lo + (hi - lo) u` with u = torch.rand(n, 1) in float64
        (the fresh points of the `resample` and `r3_continuous` samplers), then cast to `dtype`."""
        u = torch.rand(n, 1, generator=generator, dtype=torch.float64)
        return (self.lo + (self.hi - self.lo) * u).to(dtype=dtype)

    def boundary(self):
        """The two end points, shape (2,) (Boni sec. 6.2.1: two points at the boundary)."""
        return np.array([self.lo, self.hi], dtype=float)

    def contains(self, points):
        """Boolean array: points (N,) or (N, 1) in the closed interval."""
        x = np.asarray(points, dtype=float).reshape(-1)
        return (x >= self.lo) & (x <= self.hi)

    def scaled(self, c):
        """The interval in the network variable z = c x (c > 0)."""
        c = _positive(c)
        return Interval(c * self.lo, c * self.hi)

    def as_config(self):
        """Configuration (and cache-key) form: the list [lo, hi]."""
        return [self.lo, self.hi]

    def label(self):
        return f"[{self.lo:g}, {self.hi:g}]"


@dataclass(frozen=True)
class Triangle:
    """The training triangle T_margin of the one-mass box, in (s1, s2) with s0 = 1:
    {s1 <= -margin, s2 >= margin, 1 + s1 - s2 >= margin}, inside the physical region s1 < 0,
    s2 > 0, 1 + s1 - s2 > 0 (Boni 6.38). Vertices and membership are those of
    `systems.box_one_mass` (`vertices`, `in_domain`). `scale` multiplies the coordinates
    (network variable z = scale (s1, s2)); data are always in the physical variables (scale 1).
    """

    margin: float
    scale: float = 1.0

    name = "triangle"
    dim = 2

    def __post_init__(self):
        object.__setattr__(self, "margin", float(self.margin))
        object.__setattr__(self, "scale", _positive(self.scale))
        _one_mass().vertices(self.margin)            # ValueError unless 0 < margin < 1/3

    def vertices(self):
        """The three vertices, shape (3, 2), in the scaled coordinates."""
        return self.scale * _one_mass().vertices(self.margin)

    def from_unit(self, u):
        """Map points u (N, 2) of the unit square onto the triangle, uniformly, by reflection:
        (u, v) with u + v > 1 becomes (1 - u, 1 - v), then P = A + u (B - A) + v (C - A).
        Every point is used (no rejection): the number of points is exact."""
        u = np.array(u, dtype=float).reshape(-1, 2)
        flip = u.sum(axis=1) > 1
        u[flip] = 1 - u[flip]
        a, b, c = self.vertices()
        return a + u[:, :1] * (b - a) + u[:, 1:] * (c - a)

    def sample(self, n, seed=None):
        """n uniform points, shape (n, 2); `seed`: int, None or numpy Generator."""
        return self.from_unit(_rng(seed).random((n, 2)))

    def sample_columns(self, n, seed=None):
        """n uniform points, shape (n, 2) (the same as `sample`: the points of a 2-dimensional domain are columns)."""
        return self.sample(n, seed)

    def sample_torch(self, n, generator, dtype=torch.float64):
        """n uniform points (n, 2) from a torch generator: u = torch.rand(n, 2) in float64, mapped by the reflection
        of `from_unit` ((u, v) with u + v > 1 becomes (1 - u, 1 - v)), then cast to `dtype`."""
        u = torch.rand(n, 2, generator=generator, dtype=torch.float64)
        u = torch.where(u.sum(dim=1, keepdim=True) > 1, 1 - u, u)
        a, b, c = torch.as_tensor(self.vertices(), dtype=torch.float64)
        return (a + u[:, :1] * (b - a) + u[:, 1:] * (c - a)).to(dtype=dtype)

    def grid(self, n):
        """A fixed triangular lattice of at most n points (n >= 3), shape (N, 2), no random numbers: the points
        A + (i/m)(B - A) + (j/m)(C - A) with i, j >= 0, i + j <= m, for the largest m with (m+1)(m+2)/2 <= n.
        Distinct points, vertices and edges included (for the residual statistics of the trainer)."""
        if n < 3:
            raise ValueError(f"a triangular lattice needs at least 3 points, got {n}")
        m = 1
        while (m + 2) * (m + 3) // 2 <= n:
            m += 1
        i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
        keep = (i + j) <= m
        i, j = i[keep] / m, j[keep] / m
        a, b, c = self.vertices()
        return a + i[:, None] * (b - a) + j[:, None] * (c - a)

    def boundary(self):
        """The three vertices, shape (3, 2): the boundary points."""
        return self.vertices()

    def contains(self, points):
        """Boolean array: points (N, 2) in the closed triangle (tolerance of `box_one_mass.in_domain`)."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        return _one_mass().in_domain(p / self.scale, self.margin)

    def scaled(self, c):
        """The triangle in the network variable z = c (s1, s2) (c > 0)."""
        return Triangle(self.margin, self.scale * _positive(c))

    def as_config(self):
        """Configuration (and cache-key) form: {name: triangle, margin: m} (and scale if not 1)."""
        out = {"name": self.name, "margin": self.margin}
        if self.scale != 1.0:
            out["scale"] = self.scale
        return out

    def label(self):
        text = f"triangle (margin {self.margin:g})"
        return text if self.scale == 1.0 else f"{text} x {self.scale:g}"


def _positive(c):
    c = float(c)
    if not (np.isfinite(c) and c > 0):
        raise ValueError(f"the scale factor must be positive, got {c}")
    return c


#: configuration name -> domain class (dict form of `data.domain`)
DOMAINS = {"triangle": Triangle}


def as_domain(value):
    """Domain from its configuration form: a Domain is returned as is, a pair [lo, hi] is an
    Interval, a dict {name: <kind>, ...} is the domain registered under that name."""
    if isinstance(value, (Interval, Triangle)):
        return value
    if isinstance(value, dict):
        params = dict(value)
        name = params.pop("name", None)
        try:
            cls = DOMAINS[name]
        except KeyError:
            raise ValueError(f"unknown domain {name!r}; available: {sorted(DOMAINS)} "
                             "(or a pair [lo, hi] for an interval)") from None
        return cls(**params)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        return Interval(*value)
    raise ValueError(f"a domain is a pair [lo, hi] or a dict {{name: ...}}, got {value!r}")
