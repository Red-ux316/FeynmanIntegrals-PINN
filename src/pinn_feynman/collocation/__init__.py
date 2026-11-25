"""Collocation points for the physics loss: base points (uniform or Dirichlet, Boni sec. 6.1.1), fresh points, R3 on the continuum."""

from .factory import COLLOCATION_SAMPLERS, build_collocation_sampler, sampler_class
from .r3 import R3ContinuousSampler
from .sampler import BasePointsSampler, DirichletCollocationSampler, ResampleSampler

__all__ = ["COLLOCATION_SAMPLERS", "BasePointsSampler", "DirichletCollocationSampler", "R3ContinuousSampler", "ResampleSampler",
           "build_collocation_sampler", "sampler_class"]
