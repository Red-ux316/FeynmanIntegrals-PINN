"""Providers of the known master integrals M_K (and of the forcing term) of the bottom-up formulation.

Two kinds, both numpy complex128 without gradient, called in the physical variables v = z / c (shape (N,) for one
variable, (N, d) for d variables, as the points of `reference_values`):

* **masters** (`kind = "masters"`): `values(v)` -> (N, len(columns), len(orders)); the system multiplies by A_UK.
  `columns`  indices (0-based, increasing) in the complete basis;
  `orders`   Laurent orders returned, contiguous and increasing;
  `leading_order`  declaration: every coefficient of order < leading_order is exactly zero (bubbles: -1);
  `cheap`    recomputing at every change of the collocation points is acceptable (closed form, constant,
             interpolated table: yes; a network, an integrator, pySecDec: no).
* **forcing** (`kind = "forcing"`): `values(v)` -> f in the physical variables, (N, n_U, L) for one variable and
  (N, d, n_U, L) for d variables, with `rows` the indices of the unknowns U and `orders` the orders of the network.

The constructor is light (nothing is loaded before the first call of `values`), deterministic, and a provider has no
state to save in a checkpoint. The real or imaginary part is chosen by the system, not by the provider.
"""

import numpy as np


class ClosedFormProvider:
    """Masters from a numpy function `v -> complex (N, len(columns), len(orders))`."""

    kind = "masters"

    def __init__(self, columns, orders, leading_order, function, cheap=True):
        self.columns = tuple(columns)
        self.orders = tuple(orders)
        self.leading_order = int(leading_order)
        self.cheap = bool(cheap)
        self._function = function
        if list(self.orders) != list(range(self.orders[0], self.orders[0] + len(self.orders))):
            raise ValueError(f"orders {self.orders} must be contiguous and increasing")

    def values(self, v):
        out = np.asarray(self._function(np.asarray(v, dtype=float)), dtype=complex)
        if out.shape[1:] != (len(self.columns), len(self.orders)):
            raise ValueError(f"the function returned shape {out.shape}, expected (N, {len(self.columns)}, "
                             f"{len(self.orders)})")
        return out


class ConstantProvider(ClosedFormProvider):
    """Masters that do not depend on the point (a tadpole): `value` complex (len(columns), len(orders))."""

    def __init__(self, columns, orders, leading_order, value):
        value = np.asarray(value, dtype=complex)

        def function(v):
            return np.broadcast_to(value, (len(v),) + value.shape).copy()

        super().__init__(columns, orders, leading_order, function, cheap=True)


class CompositeProvider:
    """One provider of masters per block of K: columns are the union, orders the union of the windows (padded with
    zeros below the leading order of each part)."""

    kind = "masters"

    def __init__(self, parts):
        self.parts = tuple(parts)
        columns = [c for p in self.parts for c in p.columns]
        if len(set(columns)) != len(columns):
            raise ValueError(f"the parts of a composite provider overlap: columns {sorted(columns)}")
        self._order = np.argsort(columns)
        self.columns = tuple(sorted(columns))
        low = min(p.orders[0] for p in self.parts)
        high = max(p.orders[-1] for p in self.parts)
        self.orders = tuple(range(low, high + 1))
        for p in self.parts:
            if p.orders[-1] < high:
                raise ValueError(f"a part stops at order {p.orders[-1]} < {high}: every part must cover the top order")
        self.leading_order = min(p.leading_order for p in self.parts)
        self.cheap = all(p.cheap for p in self.parts)

    def values(self, v):
        blocks = []
        for p in self.parts:
            part = p.values(v)
            padded = np.zeros(part.shape[:2] + (len(self.orders),), dtype=complex)
            first = p.orders[0] - self.orders[0]
            padded[:, :, first:first + part.shape[2]] = part
            blocks.append(padded)
        return np.concatenate(blocks, axis=1)[:, self._order]


class ClosedFormForcing:
    """Forcing term from a numpy function `v -> complex (N, len(rows), len(orders))` (one variable), when the
    subtopologies enter only through a known inhomogeneous term (non-planar three-point: vMT eq. 4.2)."""

    kind = "forcing"

    def __init__(self, rows, orders, function, cheap=True):
        self.rows = tuple(rows)
        self.orders = tuple(orders)
        self.cheap = bool(cheap)
        self._function = function
        if list(self.orders) != list(range(self.orders[0], self.orders[0] + len(self.orders))):
            raise ValueError(f"orders {self.orders} must be contiguous and increasing")

    def values(self, v):
        out = np.asarray(self._function(np.asarray(v, dtype=float)), dtype=complex)
        if out.shape[1:] != (len(self.rows), len(self.orders)):
            raise ValueError(f"the function returned shape {out.shape}, expected (N, {len(self.rows)}, "
                             f"{len(self.orders)})")
        return out


class ForcingFromMasters:
    """Forcing provider built from a masters provider and the orders of A_UK, to test the forcing path of the
    system against the masters path (d = 1 or d > 1).

    `a_uk(v)` -> array (N, n_U, n_K, p + 1) for one variable, (N, d, n_U, n_K, p + 1) for d variables, in the
    physical variables: f_k^(l) = sum_j A_j,UK,k M_K^(l-j).
    """

    kind = "forcing"

    def __init__(self, masters, rows, orders, a_uk):
        self.masters, self.rows, self.orders, self._a_uk = masters, tuple(rows), tuple(orders), a_uk
        self.cheap = masters.cheap

    def values(self, v):
        mk = self.masters.values(v)                                       # (N, n_K, L_p)
        a = np.asarray(self._a_uk(v), dtype=float)
        offset = self.orders[0] - self.masters.orders[0]
        out = []
        for i in range(len(self.orders)):
            f = 0
            for j in range(a.shape[-1]):
                k = offset + i - j
                if 0 <= k < mk.shape[2]:
                    spec = "nuk,nk->nu" if a.ndim == 4 else "nduk,nk->ndu"
                    f = f + np.einsum(spec, a[..., j], mk[:, :, k])
            out.append(f)
        return np.stack(out, axis=-1)
