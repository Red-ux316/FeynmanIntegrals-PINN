"""Point weighting from the optional `point_weighting` section of a configuration."""

from importlib.util import find_spec

from .distance import DistanceWeighting
from .point_weights import NoWeighting

POINT_WEIGHTING = {"none": NoWeighting, "distance": DistanceWeighting}
if find_spec(f"{__package__}.extra") is not None:             # optional weightings, when the module is present
    from .extra import EXTRA_WEIGHTING
    POINT_WEIGHTING.update(EXTRA_WEIGHTING)


def point_weighting_class(config):
    """Class named in an optional `point_weighting` section (None: Boni's plain mean)."""
    name = (config or {"name": "none"})["name"]
    try:
        return POINT_WEIGHTING[name]
    except KeyError:
        raise ValueError(f"unknown point weighting {name!r}; available: {sorted(POINT_WEIGHTING)}") from None


def build_point_weighting(config, n_base, domain, system=None, input_scale=1.0):
    """`config` = {name: none | distance, <parameters>} or None (Boni: none).

    Each class takes what it needs from `n_base` (number of base points), `domain` (network variable), `system` (name
    of the problem) and `input_scale`: the distance weight uses the last three.
    """
    cls = point_weighting_class(config)
    params = dict(config or {"name": "none"})
    params.pop("name")
    return cls.build(params, n_base=n_base, domain=domain, system=system, input_scale=input_scale)
