# visualization/

Every figure of the project: pure functions, data in, PNG out, one style. A function takes arrays or history rows,
draws, saves the PNG and returns False when there is nothing to draw.

| Module | Contents |
|---|---|
| `style.py` | colours (`SERIES`, inks), matplotlib settings (`STYLE`), `save`/`_save`, `order_label`, `collect` (figures returned instead of written), `quiet` (silences two harmless warnings) |
| `training.py` | along the epochs (optional `smooth`: moving average over that many epochs, geometric on log axes, `moving_average`): `loss_training` (L_data, L_PDE, ReLoBRaLo-weighted sum), `loss_validation` (L_data and L_PDE on the validation points), `relobralo` (the two coefficients, their ratio, the share of the L_PDE term), `gradients` (gradient norm of the weighted L_data and L_PDE, their ratio, the ratio of the Adam steps each term alone would cause, the cosine between the two gradients; only for runs with `training.diagnostics.gradients`), `learning_rate`, `r3_retained`, `pde_sampling_bias` (training over validation L_PDE); `loss_curves` (Boni's Fig. 6.2), `training_dynamics`, `compare_val_loss`, `adaptive_columns` |
| `components.py` | per component M_i^(k): `component_grid` (one panel per master x order, exact in black), `block_curves` and `per_order_blocks` (a quantity along linear x, every seed a line; `shade` = grey spans), `per_order_cells` (a quantity per cell, one panel per order, one colour per master), `cell_curves` (several per-cell quantities in one panel); linear axis in abs(x) by default |
| `triangle.py` | maps on the triangle of the one-mass box: `order_maps` (rows = parts, columns = orders, one colour scale per row), `cell_maps` (a grid of maps on small equal triangles, `Blues` on a log scale, one scale per row or `shared`), `ratio_maps` (a ratio around 1: log10 with the diverging `RdBu_r` centred at 0), `edge_curves` and `cut_curves` (a quantity against the edge distance / along the three medians, linear axis, thin seeds and a thick geometric mean) |
| `accuracy.py` | network against the exact solution: `solution_curves`, `error_vs_x`, `error_vs_distance`, `component_heatmap`, `error_histograms`, `real_vs_imag`, and the comparison versions; for several variables `error_map` (mean absolute error on the domain, one panel per order, log colour) and `error_vs_edge_distance` (error against the distance from each edge), and for groups `compare_error_maps` and `compare_error_vs_edge_distance`; `xlabel` and `master_labels` replace the default labels (used for the sunrise, from its `FIGURE_LABELS`) |
| `singularities.py` | the singular points of each problem in one place, read from the system modules (`SINGULAR_POINTS`, `FORCING_BRANCH_POINTS`; two variables: `DENOMINATOR_FACTORS` at s0 = 1) with a label and a kind (`physical` dark red, `spurious` grey-blue, `KINDS`, `LINE_KINDS`): `singular_points(problem, scale=, absolute=, sign=)` -> [(x, label, kind)], `singular_lines(problem)` -> [(a, b, c, label, kind)] for a v1 + b v2 = c, `mark(ax, points)` (dashed vertical lines; points outside the x range, or <= 0 on a log axis, skipped; limits unchanged), `mark_lines(ax, lines)` (on a map, clipped) |
| `methods.py` | colours: `METHOD_COLOUR` (one fixed colour per method), `group_styles(labels, reference)` (**one colour per group**: the tone of the method family, `TINTS`; the reference dark grey, `set_reference`; always a solid line), `group_markers` (one marker per group), `PART_COLOUR` (real and imaginary part) |
| `tuning.py` | tuning studies: `tuning_history`, `importance_bars`, `slices`, `parallel_coordinates`, `contour_pair`, `category_boxes` |
| `data_io.py` | `save_data`/`load_data` (nested dicts of arrays in one npz), `slim` (drops per-point arrays), `history_columns`/`history_rows` |

Conventions: `groups` = {label: [data of seed 0, of seed 1, ...]}; several seeds are thin lines plus a thick median; the
colour belongs to the method; one y axis per panel.

`evaluation/plots.py` and `tuning/figures.py` re-export some of these functions. The figures of a trained run are
drawn by `python -m pinn_feynman.evaluation.evaluate <run>`.

