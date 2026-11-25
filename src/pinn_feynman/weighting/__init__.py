"""Per-point weights of the physics loss (distance from the singular points); default: none."""

from .distance import DistanceWeighting
from .factory import POINT_WEIGHTING, build_point_weighting, point_weighting_class
from .point_weights import NoWeighting

__all__ = ["POINT_WEIGHTING", "DistanceWeighting", "NoWeighting", "build_point_weighting", "point_weighting_class"]
