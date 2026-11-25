# frobenius/

Frobenius expansion of the massless box around $x=0$, in Boni's basis: the local solution $M=H(x)\,x^{\Lambda}\vec d$, with $H=\mathrm{diag}(1,1,-x)\,\tilde U(x)$, and its split into an analytic part and a part that carries the logarithms.

| File | Content |
|---|---|
| [mbox_x0.py](mbox_x0.py) | the module, in four steps: `frobenius_series` ($\Lambda$ and the coefficients of $\tilde U$ from the recursion), `fix_constants` ($\vec d$ from the exact values at a point $x_0$), `local_m3` (third row of $H\,x^{\Lambda}\vec d$ and its derivative, whole function or analytic part), `exact_m3` (closed form and $\mathrm dM_3/\mathrm dx$ from the system); `parts` builds the data of the figure. The exact box, the matrix $A$ and Cauchy's formula for the orders in $\varepsilon$ are those of the package (`reference.box_massless`, `systems.box_massless`) |
| [plot_parts.py](plot_parts.py) | $M_3^{(k)}$ and $\mathrm dM_3^{(k)}/\mathrm dx$, $k=-1,0,1$, on $[-0.6,0)$: true value, analytic part, singular part → `figures/parts_x0.pdf`, `figures/parts_x0_derivative.pdf` |
| [plot_series_error.py](plot_series_error.py) | relative difference between the local form with $\tilde U$ truncated to $N$ coefficients and the exact box at $\varepsilon=0.3$, on $-10^{-1}\ge x\ge-10^{-9}$ → `figures/series_error.pdf` |

The exact box comes from the package (`pinn_feynman.reference.box_massless`), so the package must be installed (see the root README). Run from the repository root:

```
python frobenius/plot_parts.py
python frobenius/plot_series_error.py
```
