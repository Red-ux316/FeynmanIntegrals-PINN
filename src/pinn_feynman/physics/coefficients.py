"""The coefficients of the cascade residual at a set of points: A_0..A_p and an optional forcing term."""

from dataclasses import dataclass

import torch

from .losses import cascade_residual


@dataclass(frozen=True)
class ResidualCoefficients:
    """What the residual dy^(l) - sum_j A_j y^(l-j) - f^(l) needs at the points, in the network variables z.

    orders   (A_0, A_1, ..., A_p), p >= 1, already divided by the input scale (shapes of `cascade_residual`)
    forcing  f, same shape as the residual; None for the complete system
    Indexing with the rows of the points gives the coefficients at those points.
    """

    orders: tuple
    forcing: torch.Tensor | None = None

    def __getitem__(self, idx):
        return ResidualCoefficients(tuple(a[idx] for a in self.orders),
                                    None if self.forcing is None else self.forcing[idx])

    def __len__(self):
        return len(self.orders[0])

    def residual(self, y, dy):
        return cascade_residual(y, dy, *self.orders, forcing=self.forcing)
