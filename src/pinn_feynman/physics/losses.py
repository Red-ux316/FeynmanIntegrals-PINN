"""Training losses of Boni's PINN (thesis sec. 6.1.2).

Shapes: values y are (N, n_masters, n_orders), orders from the lowest (eps^-2) up. With one variable the
derivatives dy are (N, n_masters, n_orders) and the matrices a0, a1 (N, n_masters, n_masters); with d > 1
variables they carry a direction axis: dy (N, d, n_masters, n_orders), a0, a1 (N, d, n_masters, n_masters),
and so does the residual (N, d, n_masters, n_orders).
"""

import torch
import torch.nn.functional as F

from .derivatives import value_and_derivative


def data_loss(prediction, target):
    """Eq. (6.11): mean squared error over points, master integrals and orders."""
    return torch.mean((prediction - target) ** 2)


def cascade_residual(y, dy, a0, a1, *higher, forcing=None):
    """Residual of eq. (6.5): dy^(l) - A0 y^(l) - A1 y^(l-1) [- A_j y^(l-j)] [- f^(l)], with y^(l-j) = 0 below the lowest order.

    `higher` = (A_2, ..., A_p) for systems of higher degree in eps; `forcing` = f, the known sub-topologies of the
    bottom-up formulation, in the shape of the residual. Without both, only the A0 and A1 terms.
    With a direction axis in dy (d > 1) the same y is used in every direction.
    """
    if dy.dim() == y.dim() + 1:
        y = y.unsqueeze(-3)
    previous = F.pad(y[..., :-1], (1, 0))
    residual = dy - a0 @ y - a1 @ previous
    for j, aj in enumerate(higher, start=2):
        if j < y.shape[-1]:                      # orders shifted out of the window contribute nothing
            residual = residual - aj @ F.pad(y[..., :-j], (j, 0))
    if forcing is not None:
        residual = residual - forcing
    return residual


def physics_loss(model, system, z):
    """Eq. (6.12): mean squared residual at the collocation points z of shape (N, d).

    The mean runs over points, masters, orders and, with d > 1, directions (one L_DE for all the equations).
    """
    y, dy = value_and_derivative(model, z)
    c = system.coefficients(z)
    return torch.mean(cascade_residual(y, dy, *c.orders, forcing=c.forcing) ** 2)


def residual_norm(residual):
    """Per-point size of the residual: root mean square over (directions,) masters and orders, shape (N,)."""
    return torch.sqrt(torch.mean(residual.reshape(len(residual), -1) ** 2, dim=1))


def weighted_physics_loss(residual, weights=None):
    """Eq. (6.12) with optional per-point weights w (N,): mean((w r)^2); None = plain mean."""
    if weights is None:
        return torch.mean(residual**2)
    return torch.mean((weights.reshape(-1, *([1] * (residual.dim() - 1))) * residual) ** 2)
