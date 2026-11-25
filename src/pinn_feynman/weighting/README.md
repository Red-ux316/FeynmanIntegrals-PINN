# pinn_feynman.weighting

**Per-point** weights in the physics loss. The default is `none`, i.e. Boni's loss, simple average.

At each step the trainer calls `point_weighting(idx, z, residual_norm)` with:
- the batch's base-point indices;
- their inputs $z$;
- the residual norms, detached from the graph.

It receives the weights $w$ of shape `(B,)`, or `None`. The physics loss becomes $\langle(w\,r)^2\rangle$ (`physics.losses.weighted_physics_loss`).

| Class | Name in the YAML | What it does | Source |
|---|---|---|---|
| `NoWeighting` (`point_weights.py`) | `none` | returns `None`: Boni's loss, unchanged | thesis eq. (6.12) |
| `DistanceWeighting(system, domain, input_scale)` (`distance.py`) | `distance` | distance weighting (DW): $w = d(v)/c$, $d$ = distance of the point $v = z/s$ (physical variables) from the nearest pole (one variable, `block_poles(system)`) or singular line (one-mass box, `block_lines(system)`) of the block $A_{UU}$ of the system; $c$ = RMS of $d$ on a fixed grid of the domain, so that the weights have mean square about 1 | — |

## `distance.py` — distance weighting (DW)

For the error $e$ of the network the residual is $r = e' - A_{UU}\,e$ (the known term of the lower sectors cancels). Near a pole $x_0$ of $A_{UU}$, $A_{UU}\sim A_0/(x-x_0)$, so the same error weighs like $1/|x-x_0|$ and the plain mean squared residual is dominated by the points next to the pole. Multiplying the residual by the distance $d$ from the nearest pole turns the equation into its Euler form, $d\,r$ of order $e$ everywhere.

- The poles are **computed** from the system: the denominators of $A_{UU}$ (the block on the columns learnt by the network, every order in $\varepsilon$), factored with sympy. Physical and spurious poles count alike; the poles of the lower block and of the mixed block $A_{UK}$ do not count, since they never multiply the error of the network.
- One variable: $d(x)$ = minimum over the real roots of the denominators.
- Two variables (one-mass box): every denominator factor must be a line $a v_1 + b v_2 = c$; $d$ = minimum over the lines of the Euclidean distance.
- Distances are measured in the physical variables $v = z/s$, not in the network variables $z$.
- The weight depends only on the position: no state, valid with fixed base points and with fresh points (`resample`, R3).

| Name | What it does |
|---|---|
| `block_poles(system)` | real poles of $A_{UU}$ of a one-variable problem |
| `block_lines(system)` | singular lines $(a, b, c)$, $a^2+b^2=1$, of a two-variable problem |
| `distance_function(system)` | $d(v)$ for points `(N, d)` in the physical variables, numpy or torch |
| `DistanceWeighting.weights(z)` | $w$ at the points $z$ of the network variables |

A domain that touches a pole is refused.

## `factory.py`

`POINT_WEIGHTING` (registry: `none`, `distance`), `point_weighting_class(config)`, `build_point_weighting(config, n_base, domain, system=None, input_scale=1.0)` from the optional `point_weighting` section of the YAML. Each class is built by its class method `build`, which takes from these arguments what it needs.

**Common methods:** `end_epoch()`, `state_dict()` and `load_state_dict()` for checkpointing.

With `distance`, the L_PDE in the history (`physics_loss`) is the **weighted** one; the trainer also writes the plain one, `physics_loss_unweighted`.

