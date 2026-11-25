"""Output mask on the orders in eps: the network computes only the active orders, the others are exact zeros."""

import copy

import torch
from torch import nn


class ActiveOrders(nn.Module):
    """Wraps a network with n_masters x len(active) outputs into one with n_masters x n_orders outputs.

    The inactive orders are exact zeros, in the value and in the derivative: the rest of the code (losses, residual,
    evaluation) sees the same (N, n_masters, n_orders) tables as without the mask. `active` are positions on the order
    axis (0 .. n_orders - 1), distinct and increasing. No parameter of its own: the parameters are those of `mlp`, whose
    state_dict keys are prefixed by `mlp.` (a masked run cannot be reloaded into an unmasked network by mistake).
    """

    def __init__(self, mlp, active, n_orders):
        super().__init__()
        active = [int(a) for a in active]
        if sorted(set(active)) != active or not active or active[0] < 0 or active[-1] >= n_orders:
            raise ValueError(f"active order positions {active} must be distinct, increasing and in [0, {n_orders})")
        if mlp.n_orders != len(active):
            raise ValueError(f"the inner network has {mlp.n_orders} orders, {len(active)} active positions given")
        self.mlp = mlp
        self.n_masters, self.n_orders = mlp.n_masters, n_orders
        self.register_buffer("active", torch.tensor(active, dtype=torch.long), persistent=False)

    def _scatter(self, y):
        """(..., n_masters, len(active)) -> (..., n_masters, n_orders) with zeros at the inactive orders."""
        out = y.new_zeros((*y.shape[:-1], self.n_orders))
        return out.index_copy(-1, self.active, y)

    def forward(self, z):
        """z of shape (N, in_features) -> (N, n_masters, n_orders)."""
        return self._scatter(self.mlp(z))

    def forward_with_derivative(self, z):
        """As MLP.forward_with_derivative, with the inactive orders set to zero in both outputs."""
        y, dy = self.mlp.forward_with_derivative(z)
        return self._scatter(y), self._scatter(dy)


def mask_orders(mlp, active):
    """ActiveOrders around a copy of `mlp` (an MLP with all the orders) keeping only the output rows of the active
    order positions: same hidden layers, last layer sliced. The masked network starts from the same weights as `mlp`."""
    active = [int(a) for a in active]
    n_masters, n_orders = mlp.n_masters, mlp.n_orders
    inner = copy.deepcopy(mlp)
    last = inner.net[-1]
    rows = torch.tensor([m * n_orders + a for m in range(n_masters) for a in active], dtype=torch.long)
    # skip_init: no random initialisation, so the global generator is left exactly as building `mlp` left it
    sliced = nn.utils.skip_init(nn.Linear, last.in_features, len(rows), dtype=last.weight.dtype,
                                device=last.weight.device)
    with torch.no_grad():
        sliced.weight.copy_(last.weight[rows])
        sliced.bias.copy_(last.bias[rows])
    inner.net[-1] = sliced
    inner.n_orders = len(active)
    return ActiveOrders(inner, active, n_orders)
