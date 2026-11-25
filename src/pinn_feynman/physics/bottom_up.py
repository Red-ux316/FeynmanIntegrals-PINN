"""Bottom-up system: the network learns the unknowns U, the knowns K come from a provider.

With M = (M_U, M_K), A_KU = 0 and A = sum_j eps^j A_j:
    dM_U/dz_k = A_UU,k M_U + f_k,    f_k^(l) = sum_j A_j,UK,k M_K^(l-j)
`BottomUpSystem` has the interface of `TorchSystem` used by the trainer (`coefficients`, `n_masters`, `variables`,
`n_variables`, `input_scale`, `cheap`); its coefficients carry the forcing f in the network variables z, computed in
float64 (no gradient) and cached by the trainer next to A_UU (recomputed when the collocation points change).

The real or imaginary part is fixed at construction: the decoupling holds when A is real in the domain (massless and
one-mass boxes, sunrise at real s and m^2); it is an assumption of the case, not a guarantee of the method.
"""

import numpy as np
import torch

from .coefficients import ResidualCoefficients

PARTS = ("real", "imag")


def _part(values, part):
    return values.real if part == "real" else values.imag


class BottomUpSystem:
    """A_UU(z) and the forcing term f(z) for the unknowns U.

    uu        `TorchSystem` of A_UU (n_U x n_U)
    uk        `TorchSystem` of A_UK (n_U x n_K, same variables and scale), or None with a forcing provider
    provider  providers.base interface, kind "masters" or "forcing"
    part      'real' or 'imag'
    orders    Laurent orders of the network (increasing, contiguous)
    unknowns  indices of U in the complete basis (checked against `provider.columns` / `provider.rows`)
    """

    def __init__(self, uu, uk, provider, part, orders, unknowns=None, knowns=None):
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}, got {part!r}")
        self.uu, self.uk, self.provider, self.part = uu, uk, provider, part
        self.orders = tuple(orders)
        self.n_masters = uu.n_masters
        self.variables, self.n_variables, self.input_scale = uu.variables, uu.n_variables, uu.input_scale
        self.cheap = bool(provider.cheap)
        if provider.kind == "masters":
            if uk is None:
                raise ValueError("a provider of masters needs A_UK")
            if knowns is not None and tuple(provider.columns) != tuple(knowns):
                raise ValueError(f"the provider gives the columns {tuple(provider.columns)}, the knowns are "
                                 f"{tuple(knowns)}")
            if uk.degree != uu.degree:
                raise ValueError(f"A_UU has degree {uu.degree} in eps, A_UK {uk.degree}: `blocks.split` gives the same")
            if (uk.variables, uk.input_scale) != (uu.variables, uu.input_scale):
                raise ValueError("A_UU and A_UK must have the same variables and input scale")
            self._check_window()
        elif provider.kind == "forcing":
            if unknowns is not None and tuple(provider.rows) != tuple(unknowns):
                raise ValueError(f"the provider gives the rows {tuple(provider.rows)}, the unknowns are "
                                 f"{tuple(unknowns)}")
            if tuple(provider.orders) != self.orders:
                raise ValueError(f"the forcing provider has orders {tuple(provider.orders)}, the network {self.orders}")
        else:
            raise ValueError(f"unknown provider kind {provider.kind!r}")

    # --- window of orders ----------------------------------------------------------------------------------
    @property
    def _shift(self):
        return self.uk.degree

    def _check_window(self):
        """M_K is needed at the orders l_min - p .. l_max: each must be returned or below the leading order."""
        have = set(self.provider.orders)
        missing = [o for o in range(self.orders[0] - self._shift, self.orders[-1] + 1)
                   if o not in have and o >= self.provider.leading_order]
        if missing:
            raise ValueError(f"the provider does not cover the orders {missing} of the knowns (it returns "
                             f"{tuple(self.provider.orders)} and is zero below {self.provider.leading_order})")

    # --- interface of TorchSystem ----------------------------------------------------------------------------
    def matrices(self, z):
        raise TypeError("bottom-up system: the residual needs its forcing, use coefficients(z)")

    def coefficients(self, z):
        return ResidualCoefficients(self.uu.coefficient_orders(z), self.forcing(z))

    def forcing(self, z):
        """f at the points z: shape of the residual, in the network variables, dtype and device of z."""
        v = (z.detach().double().cpu() / self.input_scale).numpy()
        v = v.reshape(-1) if self.n_variables == 1 else v
        if self.provider.kind == "forcing":
            f = _part(self.provider.values(v), self.part) / self.input_scale
            return torch.as_tensor(np.ascontiguousarray(f), dtype=z.dtype, device=z.device)
        mk = self._known_masters(v)                                    # (N, n_K, L + p) real
        zd = z.detach().double().cpu()
        a = self.uk.coefficient_orders(zd)                             # each (N[, d], n_U, n_K), already / c
        n_orders, p = len(self.orders), self._shift
        mk = torch.as_tensor(mk)
        if self.n_variables > 1:
            mk = mk.unsqueeze(1)
        f = 0
        for j in range(p + 1):
            f = f + a[j] @ mk[..., p - j:p - j + n_orders]
        return f.to(dtype=z.dtype, device=z.device)

    def _known_masters(self, v):
        """Real parts of M_K at the orders l_min - p .. l_max, zero below the leading order of the provider."""
        values = _part(self.provider.values(v), self.part)
        first = self.orders[0] - self._shift
        out = np.zeros(values.shape[:2] + (len(self.orders) + self._shift,))
        for index, order in enumerate(range(first, self.orders[-1] + 1)):
            if order in self.provider.orders:
                out[:, :, index] = values[:, :, self.provider.orders.index(order)]
        return out
