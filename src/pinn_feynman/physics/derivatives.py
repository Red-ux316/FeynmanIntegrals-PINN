"""Derivatives of the network output with respect to its inputs, one per direction."""

import torch


def value_and_derivative(model, z, fast=True):
    """(model(z), derivatives of model(z)) for inputs z of shape (N, d).

    d = 1: the derivative d model/dz, shape (N, n_masters, n_orders) like the output.
    d > 1: the derivatives along every input, shape (N, d, n_masters, n_orders), axis 1 = direction k.

    If the model provides `forward_with_derivative` (hand-written forward mode, see models/mlp.py) and `fast` is
    true, that is used. Otherwise torch.func.jvp with one tangent per direction (the unit vector e_k): one
    Jacobian-vector product gives the derivative of every output along e_k. Both stay differentiable with respect
    to the model parameters.
    """
    if fast and hasattr(model, "forward_with_derivative"):
        return model.forward_with_derivative(z)
    if z.shape[-1] == 1:
        return torch.func.jvp(model, (z,), (torch.ones_like(z),))
    y, derivatives = None, []
    for k in range(z.shape[-1]):
        tangent = torch.zeros_like(z)
        tangent[..., k] = 1
        y, dy = torch.func.jvp(model, (z,), (tangent,))
        derivatives.append(dy)
    return y, torch.stack(derivatives, dim=1)
