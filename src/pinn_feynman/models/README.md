# pinn_feynman.models

Neural networks. The output always has shape `(N, n_masters, n_orders)`: Master Integral index, then the index of the order in $\varepsilon$ from the lowest ($\varepsilon^{-2}$) upward, as in the datasets.

## `mlp.py`

| Name | What it does |
|---|---|
| `MLP(in_features, hidden, n_masters, n_orders, activation)` | multilayer perceptron; `forward(z)` → `(N, n_masters, n_orders)` |
| `MLP.forward_with_derivative(z)` | value **and derivative** with respect to the scalar input, propagating the tangent layer by layer (linear: $W\,dh$; activation: $\sigma'(h)\,dh$). Gives the same result as `torch.func.jvp` (test). Speed (torch 2.14, CPU, one thread, $N=65536$): for one input variable it is *slower* than `jvp`, 0.5 s against 0.35 s; for two input variables it is faster. With several input variables ($d>1$) the tangents have shape `(N, d, width)`; for $d=1$ there is a single tangent per point |
| `ACTIVATION_DERIVATIVES` | exact derivatives of GELU ($\Phi(h)+h\varphi(h)$), tanh, SiLU |

Boni's configuration (Tab. 6.1): input 1, hidden layers `[32, 32, 32]`, GELU, output 15 = 3 MI × 5 orders: 2671 parameters.

## `factory.py`

| Name | What it does |
|---|---|
| `MODELS`, `ACTIVATIONS` | registry by name (`mlp`; `gelu`, `tanh`, `silu`) |
| `build_model(model_config, n_masters, n_orders, dtype, seed, orders=None)` | network from the `model` section of the YAML; the number of outputs is derived from the problem, not configured; `seed` fixes the initial weights (PyTorch's default initialization). Without `model.active_orders` (or with it `null`) the result is the plain `MLP`; with it, `orders` (the order labels of the problem, e.g. `(0, 1, 2, 3, 4)`) is required and the result is an `ActiveOrders` (or the plain `MLP` when the key lists every order) |
| `active_positions(active_orders, orders)` | positions on the order axis of the labels `active_orders`; `ValueError` for labels not among `orders`, duplicates, an empty or non-increasing list |
| `count_parameters(model)` | number of parameters |

## `order_mask.py`

Optional output mask on the orders in $\varepsilon$, set by `model.active_orders` (labels of the orders the network computes, e.g. `[4]` for the non-planar three-point function, whose orders $\varepsilon^0..\varepsilon^3$ vanish identically). The other orders are **exact zeros**, in the value and in the derivative; the rest of the code sees the usual `(N, n_masters, n_orders)` tables.

| Name | What it does |
|---|---|
| `ActiveOrders(mlp, active, n_orders)` | wraps an `MLP` with `n_masters x len(active)` outputs (in `self.mlp`); `forward(z)` and `forward_with_derivative(z)` scatter its outputs to the positions `active` of an all-zero `(N, n_masters, n_orders)` table (`(N, d, n_masters, n_orders)` for the derivative with $d>1$ inputs). No parameters of its own; `state_dict` keys are `mlp.net.…`, so a masked run cannot be reloaded without the mask (keys differ: explicit error) |
| `mask_orders(mlp, active)` | `ActiveOrders` around a copy of a full `MLP`, whose last layer keeps only the rows `master * n_orders + a` of the active positions `a` (weights copied, `skip_init`: no random draw). `build_model` builds the full network with the usual random draws first, then slices it: the masked network has the same initial weights on the active orders, and leaves the global generator in the same state, as the unmasked one with the same seed |

`training.setup.check_active_orders` refuses a mask that leaves out an order the physics does not set to zero (`NONZERO_ORDERS` of the system module; all the orders when the module declares none).

