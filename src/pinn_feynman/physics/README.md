# pinn_feynman.physics

The "physics" part of the PINN: connection matrix in torch, network derivatives, data loss and physics loss (Boni, thesis §6.1.2).

## Network variable
The systems in `pinn_feynman.systems` are written in $x=t/s$. The network can use a rescaled variable $z=c\,x$: Boni uses $x_B=10\,t/s$. By the chain rule
$$\frac{d\vec M}{dz}=\frac1c\,A\!\left(\frac zc\right)\vec M .$$
The MI values at the points do not change: only the variable with respect to which the derivative is taken changes.

## One or several variables
The same code serves every problem. A system has $d$ variables (`VARIABLES` of its `systems` module): the massless box has one ($x$), the one-mass box two ($s_1,s_2$). **The direction axis exists only for $d>1$**: for $d=1$ the shapes are `(N, M, M)` (matrices), `(N, M, L)` (values, derivatives, residual); for $d>1$ they are `(N, d, M, M)` and `(N, d, M, L)`, the axis 1 being the direction. Boni (6.12) has a single $L_{DE}$ that averages the squared residual over points, directions, master integrals and orders. `residual_norm` (per point) is the RMS over directions, master integrals and orders, consistent with it. The scale $z=c\,x$ is one scalar for all the variables. The training pipeline accepts $d>1$ (`training.setup.check_variables`, `check_multi_variable`): the recorder, zones and reports are one-dimensional and are refused with an explicit error.

## `system.py`

| Name | What it does |
|---|---|
| `SYSTEMS` | view of `pinn_feynman.problems`: problem → `systems` module (`box_massless`, `box_one_mass`, `sunrise_equal_mass`, and their `*_bu` variants) |
| `system_variables(name)` | the variables of a registered problem, e.g. `("x",)` or `("s1", "s2")` |
| `TorchSystem(a0, a1, symbol, input_scale)` | one variable: the sympy matrices $A^{(0)}$, $A^{(1)}$ turned into torch functions (`lambdify`, rational functions only) |
| `TorchSystem.from_orders(orders, symbols, input_scale, variables)`, `TorchSystem.from_module(module, input_scale)` | several variables: one pair $(A^{(0)}_k, A^{(1)}_k)$ per direction $k$ (`orders`), with the symbols in `symbols`. Attributes `n_variables` and `variables` |
| `TorchSystem.matrices(z)` | $(A^{(0)}_z, A^{(1)}_z)$ at the points `z`: shape `(N, n, n)` for $d=1$ (`z` of shape `(N, 1)` or `(N,)`), `(N, d, n, n)` for $d>1$ (`z` must be `(N, d)`; a shape `(N,)` is ambiguous and is refused) |
| `TorchSystem.degree`, `.n_columns`, `.cheap` | degree in $\varepsilon$ ($p\ge1$; matrices of lower degree are completed with zeros), number of columns (rectangular blocks such as $A_{UK}$ are allowed), `cheap = True` |
| `TorchSystem.coefficient_orders(z)`, `.coefficients(z)` | $(A_0,\dots,A_p)$ at the points, and the `ResidualCoefficients` (no forcing) that the residual consumers use. `matrices(z)` is kept for degree 1 and raises for higher degree |
| `build_system(name, input_scale, part=None)` | `TorchSystem` for a registered problem; for a bottom-up problem (`unknowns` and `provider`) a `BottomUpSystem` for `part` (`'real'`/`'imag'`, required) |

## `coefficients.py` and `bottom_up.py` — bottom-up formulation

The network learns only the unknowns $U$; the known master integrals $K$ (sub-topologies) come from a provider (`pinn_feynman.providers`) and enter as a forcing term: with $A_{KU}=0$, $\partial_{z_k}M_U=A_{UU,k}M_U+f_k$, $f_k^{(l)}=\sum_jA_{j,UK,k}M_K^{(l-j)}$ (the cascade is in $\varepsilon$ to any degree $p$). The split into $U$ and $K$ and the check $A_{KU}=0$ are `systems.blocks.split`.

| Name | What it does |
|---|---|
| `ResidualCoefficients(orders, forcing)` | $(A_0,\dots,A_p)$ and the optional forcing $f$ at a set of points, already in the network variables; indexing with point rows gives the coefficients at those points. The trainer caches it (`coefficients_base`, `_grid`, `_val`; `a0_base`, ... are read-only properties) |
| `BottomUpSystem(uu, uk, provider, part, orders, unknowns, knowns)` | `coefficients(z)` = $A_{UU}$ orders + forcing $f$ computed in float64 without gradient (provider called in $v=z/c$, factor $1/c$ included); `matrices(z)` raises `TypeError`. At construction it checks the window of orders ($M_K$ is needed at $l_{\min}-p\ldots l_{\max}$: returned by the provider or below its `leading_order`) and the columns. `cheap` = the provider's |

The real/imaginary decoupling holds because $A$ is real in the domain (massless, one-mass): an assumption of the case.

## `derivatives.py`

| Name | What it does |
|---|---|
| `value_and_derivative(model, z, fast=True)` | $(y, dy/dz)$ for $d=1$; for $d>1$ the derivatives along every direction, shape `(N, d, M, L)`. Uses `model.forward_with_derivative` if it exists; otherwise `torch.func.jvp` with the tangent $e_k$ (one pass per direction) |

## `losses.py`

| Name | Equation | What it does |
|---|---|---|
| `data_loss(prediction, target)` | (6.11) | MSE over points, MI and orders |
| `cascade_residual(y, dy, a0, a1, *higher, forcing=None)` | (6.5) | $\partial_z y^{(l)}-A^{(0)}y^{(l)}-A^{(1)}y^{(l-1)}$, with $y^{(l-1)}=0$ below the lowest order; `higher` = $A_2,\dots$, `forcing` = $f$. Without them only the $A^{(0)}$ and $A^{(1)}$ terms |
| `physics_loss(model, system, z)` | (6.12) | MSE of the residual at the collocation points, averaged over points × directions × MI × orders ($n_s=1$ for the massless box, $n_s=2$ for the one-mass box) |

## `propagation.py`

| Name | What it does |
|---|---|
| `propagation_stats(norms)` | mean, max, skewness (corrected Fisher-Pearson) and kurtosis (Fisher) of the residual norms: the propagation-failure signals of Daw et al. (arXiv:2207.02338, App. H). Used after training (`evaluation.diagnostics`) and at every epoch (`Trainer`) |

**Checks**:
1. `TorchSystem` matches the numpy version in `systems`; the rescaling gives exactly $A(x_B/10)/10$, computed by hand;
2. the residual is **zero on the true values**, real and imaginary parts, in $x$ and in $x_B$. The exact derivatives are computed with `mpmath.diff` on the closed formulas, hence without using the differential equation;
3. a wrong cascade (term $A^{(1)}$ on the same order) gives a nonzero residual;
4. the forward-mode derivatives (fast and `jvp`) match the reverse-mode ones, also for the gradients with respect to the weights, for GELU, tanh and SiLU;
5. the data loss is an MSE; the physics loss is differentiable.

**Checks of several directions**:
1. systems with one variable keep the shapes `(N, M, M)` and `(N, M, L)`;
2. fast derivative = `jvp` = `jacfwd` = reverse mode, $d=1$ and $d=2$;
3. the torch matrices of the one-mass box equal `systems.box_one_mass.numeric_orders`, with and without the scale;
4. **integration test:** the true values of the one-mass box (fast method of `reference`) differentiated numerically in $s_1$ and $s_2$ (5-point scheme, $h=10^{-3}$ and $2\cdot10^{-3}$) give a residual that scales like $h^4$ and is about $10^{-7}$ of $|AM|$ at 23 points of the triangle with margin 0.1 (vertices included), real and imaginary part, with the input scale 10; with the directions swapped the residual exceeds $10^{-2}$ of $|AM|$ (the test sees the difference);
5. $L_{DE}$ is the mean over directions (hand-computed case); clear errors for an unknown system or variable, for $z$ of shape `(N,)` with $d>1$, and for $d>1$ in the trainer.
