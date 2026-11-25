"""Point samplers on a domain (`data.domains`: interval or triangle), selected by name.

`domain` may be a domain object or its configuration form (a pair (lo, hi) is an interval).
"""

from .domains import as_domain
from .interpolation import chebyshev_nodes


def boundary_points(domain, n=2, seed=None):
    """The boundary points of the domain: the two end points of an interval (Boni sec. 6.2.1:
    two points at the boundary), the three vertices of the one-mass triangle."""
    points = as_domain(domain).boundary()
    if n != len(points):
        raise ValueError(f"this domain has exactly {len(points)} boundary points, got n = {n}")
    return points


def uniform_points(domain, n, seed):
    """n points uniformly distributed in the domain, reproducible from the seed;
    shape (n,) on an interval, (n, d) otherwise."""
    return as_domain(domain).sample(n, seed)


def chebyshev_points(domain, n, seed=None):
    """n Chebyshev points of the second kind (interpolation nodes); one variable only."""
    domain = as_domain(domain)
    if domain.dim != 1:
        raise ValueError(f"Chebyshev nodes are one-dimensional: not available on {domain.label()}")
    return chebyshev_nodes(domain, n)


SAMPLERS = {
    "boundary": boundary_points,
    "uniform": uniform_points,
    "chebyshev": chebyshev_points,
}


def sample(name, domain, n, seed):
    """Dispatch to the sampler registered under `name`."""
    try:
        sampler = SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}") from None
    return sampler(domain, n=n, seed=seed)
