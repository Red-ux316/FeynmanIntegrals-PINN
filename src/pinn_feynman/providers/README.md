# pinn_feynman.providers

Providers of the **known** master integrals $M_K$ (the sub-topologies) for the bottom-up formulation (`physics/README.md`): the network learns the top block $M_U$, the provider gives the rest as a forcing term. Numpy, complex128, no gradient, called in the physical variables $v=z/c$ (`(N,)` for one variable, `(N, d)` for several). The real or imaginary part is chosen by the system.

Interface (`base.py`): `kind = "masters"` with `columns`, `orders` (contiguous), `leading_order` (exact zero below it), `cheap` (recomputation at every change of points is acceptable; checked by `training.setup.check_provider_cost`) and `values(v)` → `(N, len(columns), len(orders))`; or `kind = "forcing"` with `rows`, `orders`, `values(v)` → $f$ in the physical variables. A light constructor, no state to checkpoint.

| Name | What it does |
|---|---|
| `ClosedFormProvider(columns, orders, leading_order, function, cheap=True)` | masters from a numpy function |
| `ConstantProvider(columns, orders, leading_order, value)` | masters independent of the point (tadpole) |
| `CompositeProvider(parts)` | one provider per block of $K$; orders padded with zeros |
| `ClosedFormForcing(rows, orders, function, cheap=True)` | forcing term from a numpy function `v -> (N, len(rows), len(orders))` (one variable): the subtopologies enter only through a known inhomogeneous term |
| `ForcingFromMasters(masters, rows, orders, a_uk)` | forcing provider built from a masters provider and $A_{UK}$: used by the tests to check the forcing path |
| `box_massless.bubbles()` | bubbles $M_1=e^{i\pi\varepsilon}B(\varepsilon)$, $M_2=|x|^\varepsilon e^{2i\pi\varepsilon}B(\varepsilon)$, columns 0, 1 |
| `box_one_mass.bubbles()` | bubbles in $t$, $m^2$, $s$ (columns 1, 2, 3), copied from `reference.box_one_mass` (it uses its private series helpers) |
| `sunrise_equal_mass.tadpole()` | product of two tadpoles $I_{11000}=3^{2-2\varepsilon}\Gamma(\varepsilon-1)^2$ (column 0), constant in $s$, orders $\varepsilon^{-2}\ldots\varepsilon^0$, `leading_order = -2`; the series is `reference.sunrise_equal_mass.tadpole_series` (mpmath on the first call). The matrix is quadratic in $\varepsilon$: the forcing at $\varepsilon^l$ uses the tadpole down to $\varepsilon^{l-2}$, zero below $\varepsilon^{-2}$ |
| `nonplanar_three_point.source()` | forcing term of the non-planar top masters $(m_{10},m_{11})$ (rows 0, 1, orders $\varepsilon^0\ldots\varepsilon^4$): zero except $N_{11}^{(4)}$ = `reference.nonplanar_three_point.forcing` (vMT 4.34–4.36, continued to $x-i0$; complex for $x<0$) |

