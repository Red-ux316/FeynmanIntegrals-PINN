"""Fully connected network: x -> Laurent coefficients of the master integrals."""

import math

import torch
from torch import nn


def _gelu_derivative(h):
    """d/dh [h Phi(h)] = Phi(h) + h phi(h) (exact GELU, torch's default)."""
    cdf = 0.5 * (1 + torch.erf(h / math.sqrt(2)))
    pdf = torch.exp(-0.5 * h * h) / math.sqrt(2 * math.pi)
    return cdf + h * pdf


def _tanh_derivative(h):
    return 1 - torch.tanh(h) ** 2


def _silu_derivative(h):
    s = torch.sigmoid(h)
    return s * (1 + h * (1 - s))


#: activation class -> its derivative (for the forward-mode fast path)
ACTIVATION_DERIVATIVES = {nn.GELU: _gelu_derivative, nn.Tanh: _tanh_derivative, nn.SiLU: _silu_derivative}


class MLP(nn.Module):
    """Multilayer perceptron with output reshaped to (N, n_masters, n_orders).

    Boni's network (Table 6.1): 1 input, hidden layers [32, 32, 32], GELU, 15 outputs
    (3 master integrals x 5 orders in eps).
    """

    def __init__(self, in_features, hidden, n_masters, n_orders, activation):
        super().__init__()
        self.n_masters, self.n_orders = n_masters, n_orders
        widths = [in_features, *hidden]
        layers = []
        for w_in, w_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(w_in, w_out), activation()]
        layers.append(nn.Linear(widths[-1], n_masters * n_orders))
        self.net = nn.Sequential(*layers)

    def _shape(self, out):
        return out.reshape(-1, self.n_masters, self.n_orders)

    def forward(self, z):
        """z of shape (N, in_features) -> (N, n_masters, n_orders)."""
        return self._shape(self.net(z))

    def forward_with_derivative(self, z):
        """(output, derivatives of the output), propagating the tangents layer by layer.

        z of shape (N, d). d = 1: derivative d output/dz, shape (N, n_masters, n_orders). d > 1: one tangent per
        input direction e_k, derivatives of shape (N, d, n_masters, n_orders), axis 1 = direction k.
        Same result as torch.func.jvp (tested); slower than jvp for one input, faster for two (see the README).
        """
        if z.shape[-1] == 1:
            # one input: a single tangent per point
            h, dh = z, torch.ones_like(z)
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    h, dh = layer(h), dh @ layer.weight.T
                else:
                    dh = dh * ACTIVATION_DERIVATIVES[type(layer)](h)
                    h = layer(h)
            return self._shape(h), self._shape(dh)
        n, d = z.shape
        h = z
        dh = torch.eye(d, dtype=z.dtype, device=z.device).expand(n, d, d)     # (N, direction, width)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h, dh = layer(h), dh @ layer.weight.T
            else:
                dh = dh * ACTIVATION_DERIVATIVES[type(layer)](h).unsqueeze(1)
                h = layer(h)
        return self._shape(h), dh.reshape(n, d, self.n_masters, self.n_orders)
