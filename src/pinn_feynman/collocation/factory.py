"""Collocation sampler from the `collocation` section of a configuration."""

from importlib.util import find_spec

from .r3 import R3ContinuousSampler
from .sampler import BasePointsSampler, ResampleSampler

COLLOCATION_SAMPLERS = {"base_points": BasePointsSampler, "resample": ResampleSampler,
                        "dirichlet": BasePointsSampler,         # base points with Dirichlet probabilities (Boni)
                        "r3_continuous": R3ContinuousSampler}
if find_spec(f"{__package__}.extra") is not None:             # optional samplers, when the module is present
    from .extra import EXTRA_SAMPLERS
    COLLOCATION_SAMPLERS.update(EXTRA_SAMPLERS)


def sampler_class(config):
    """Class of the sampler named in a `collocation` section (to check combinations before building)."""
    name = config["name"]
    try:
        return COLLOCATION_SAMPLERS[name]
    except KeyError:
        raise ValueError(f"unknown collocation sampler {name!r}; available: {sorted(COLLOCATION_SAMPLERS)}") from None


def build_collocation_sampler(config, domain, seed=None, dtype=None, gate=None):
    """Sampler on `domain` (network variable); `config` = {name: ..., <parameters>}.

    `dirichlet` is `base_points` with Dirichlet probabilities (Boni's set-up, concentration 1 by default). A sampler
    whose class has `needs_gate` gets the causal gate of the point weighting, shared with the loss.
    """
    cls = sampler_class(config)
    params = dict(config)
    name = params.pop("name")
    if name == "dirichlet":
        params["probabilities"] = "dirichlet"
        params.setdefault("concentration", 1.0)
    params.setdefault("seed", seed)
    if dtype is not None:
        params["dtype"] = dtype
    if getattr(cls, "needs_gate", False):
        if gate is None:
            raise ValueError(f"{name} needs point_weighting: {{name: causal}} (the gate is shared)")
        params["gate"] = gate
    return cls(domain, **params)
