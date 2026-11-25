"""Loss balancer from the `balancing` section of a configuration."""

from .fixed import FixedWeights
from .relobralo import ReLoBRaLo

BALANCERS = {"relobralo": ReLoBRaLo, "fixed": FixedWeights}


def build_balancer(config, n_losses, seed=None):
    """Balancer for n_losses terms; `config` = {name: ..., <parameters of the class>}."""
    params = dict(config)
    name = params.pop("name")
    try:
        cls = BALANCERS[name]
    except KeyError:
        raise ValueError(f"unknown balancer {name!r}; available: {sorted(BALANCERS)}") from None
    if cls is ReLoBRaLo:
        params.setdefault("seed", seed)
    return cls(n_losses, **params)
