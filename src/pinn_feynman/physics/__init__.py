"""Differential-equation side of the PINN: connection matrix, derivatives, losses."""

from .derivatives import value_and_derivative
from .coefficients import ResidualCoefficients
from .losses import cascade_residual, data_loss, physics_loss, residual_norm, weighted_physics_loss
from .propagation import propagation_stats
from .system import SYSTEMS, TorchSystem, build_system, system_variables

__all__ = ["ResidualCoefficients", "SYSTEMS", "TorchSystem", "build_system", "system_variables", "cascade_residual", "data_loss",
           "physics_loss", "residual_norm",
           "weighted_physics_loss", "propagation_stats",
           "value_and_derivative"]
