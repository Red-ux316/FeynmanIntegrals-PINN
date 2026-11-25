# pinn_feynman.evaluation

Evaluation of the trained networks on the test set: Boni's errors (thesis eq. 6.18–6.19), figures like those in the thesis (Fig. 6.2–6.7), comparison with the numbers the thesis reports.

## Modules

| Module | Name | What it does |
|---|---|---|
| `metrics.py` | `absolute_error(pred, target)` | (6.18) $\lvert y_{nn}-y_{test}\rvert$ element by element |
| | `relative_error(pred, target, threshold)` | (6.18) $\lvert y_{nn}-y_{test}\rvert/\lvert y_{test}\rvert$; `NaN` where $\lvert y_{test}\rvert<10^{-10}$ (p. 98) |
| | `summarize(errors, orders)` | mean, median, 95th percentile, maximum, count: `overall`, `by_order` and `by_master` (6.19), `by_component` (masters × orders table) |
| `predict.py` | `load_run(dir, which)` → `Run` | configuration, part (`real`/`imag`), `best` or `final` network, `summary.json`; `Run.problem`, `Run.variables` |
| | `problem_shape(problem)` | `(number of masters, orders as a tuple, variables)`, read from what is registered (masters from `physics.SYSTEMS`, variables from `system_variables`, orders from the reference generator of `data.generation`), checked against `N_MASTERS` of the reference module |
| | `predict(model, x, input_scale)` | network output at the points ($x=t/s$ for the massless box, shape `(N,)`; $(s_1,s_2)$ for the one-mass box, shape `(N,2)`), in batches |
| | `require_one_variable(config, what)` | `ValueError` with a clear message when a one-variable tool is used on a problem with more variables |
| `plots.py` | `loss_curves`, `error_histograms`, `real_vs_imag`, `solution_curves`, `component_heatmap`, `training_dynamics`, `error_vs_distance`, `error_vs_x`, ... | the figures of `visualization/` used by `evaluate.py`, re-exported: Fig. 6.2 (training and validation loss), Fig. 6.3 (histograms, one panel per order, logarithmic bins, run mean solid and Boni's mean dashed), Fig. 6.4 (real and imaginary parts), Fig. 6.5–6.7 (network against the exact solution), the error per master and order, and the error along the domain and against the distance from the singular point |
| `boni_reference.py` | `BONI_MASSLESS_BOX`, `BONI_ONE_MASS_BOX`, `BONI_NONPLANAR` | the numbers stated in Boni's thesis (epochs, $L_{train}$, $L_{val}$, mean absolute and relative error), with the page of each |
| | `BONI`, `boni_part(problem, part)` | the numbers of a problem (from the `boni` it declares in `problems/`), `{}` if none |
| `evaluate.py` | `python -m pinn_feynman.evaluation.evaluate <run> [<run> ...] [--dataset test] [--which best]` | `evaluation.json` and figures in each run; with one real and one imaginary run, also `real_vs_imag.png` in the parent folder; the figures along x carry dashed lines at the singular points inside the plotted range (`visualization/singularities.py`) |

## One code for one and two variables
`evaluate.py` serves both. For $d=1$ (massless box) the figures are the ones above. For $d>1$ (one-mass box): losses, dynamics, component table with the names of the masters, the two histograms (with the numbers of Boni), `error_map.png` (mean absolute error over the masters, one panel per order, hexagonal cells with the mean, one common logarithmic colour scale, outline of the triangle, axes $s_1=t/s$, $s_2=m^2/s$) and `error_vs_edge_distance.png` (one column per edge $t=0$, $m^2=0$, $u=0$, each point assigned to its nearest edge; absolute error above, relative below). The curves along $x$ exist only for one variable. `real_vs_imag.png` (Fig. 6.11) only if the two parts are evaluated in the same command. Many one-mass components are identically zero (real part: order $\varepsilon^{-2}$ of masters 2-4; imaginary part: master 2 and orders $\varepsilon^{-2},\varepsilon^{-1}$ of all masters) and master 3 is constant: absolute errors count them, relative errors exclude them, so the means differ; read the median too.

Figures: PNG on a light background, categorical palette in fixed order, text in neutral inks, a single $y$ axis per panel.

**Note on the comparison with Boni.** The thesis speaks of "mean" on log-scale histograms without saying which mean. We report the arithmetic mean, the median and the 95th percentile.

