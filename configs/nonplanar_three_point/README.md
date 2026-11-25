# configs/nonplanar_three_point/

Configurations of the non-planar two-loop three-point function with a massive loop
(Boni §6.3.3): the two masters of the top sector in the basis of von Manteuffel–Tancredi (arXiv:1701.05905, eq. 2.3),
orders $\varepsilon^0\ldots\varepsilon^4$; here only $\varepsilon^4$ is trained (`model.active_orders`), the only
non-zero order. The subtopologies enter only through the closed-form forcing term (the problem is bottom-up by
nature). Variable $x=-s/m^2$, network input $x$ itself (`physics.input_scale: 1`); the domains lie on either side of
the threshold $x=-4$ without crossing it (singular points $x=-4$ and $x=0$). For $x<0$ the masters are complex: both
parts are trained.

Each file starts from its base (`base:`) and writes only what changes; the table lists those changes
(`python -m pinn_feynman.devtools.list_configs nonplanar_three_point` prints the same tree).

| File | Base | What it sets (with respect to its base) |
|---|---|---|
| `baseline.yaml` | — | complete configuration |
| `bu_baseline.yaml` | `baseline.yaml` | `physics.system: nonplanar_three_point_bu` |
| `bu_no_pole.yaml` | `bu_baseline.yaml` | `data.domain: [-20.0, -0.1]` |
| `bu_no_pole_0001.yaml` | `bu_no_pole.yaml` | `data.domain: [-20.0, -0.001]`; `training.diagnostics.gradients.every: 64` |
| `bu_no_pole_0001_exp.yaml` | `bu_no_pole_0001.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_no_pole_0001_exp_eps4.yaml` | `bu_no_pole_0001_exp.yaml` | `model.active_orders: [4]` |
| `bu_thr_left_001_exp_eps4.yaml` | `bu_no_pole_0001_exp_eps4.yaml` | `data.domain: [-7.0, -4.01]` |
| `bu_thr_left_001_exp_eps4_p3.yaml` | `bu_thr_left_001_exp_eps4.yaml` | `point_weighting.name: distance` |
| `bu_thr_left_001_exp_r3_continuous_eps4.yaml` | `bu_thr_left_001_exp_eps4.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128` |
| `bu_thr_left_01_exp_eps4.yaml` | `bu_no_pole_0001_exp_eps4.yaml` | `data.domain: [-7.0, -4.1]` |
| `bu_thr_left_01_exp_eps4_p3.yaml` | `bu_thr_left_01_exp_eps4.yaml` | `point_weighting.name: distance` |
| `bu_thr_left_01_exp_r3_continuous_eps4.yaml` | `bu_thr_left_01_exp_eps4.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128` |
| `bu_thr_right_001_exp_eps4.yaml` | `bu_no_pole_0001_exp_eps4.yaml` | `data.domain: [-3.99, -0.01]` |
| `bu_thr_right_001_exp_eps4_p3.yaml` | `bu_thr_right_001_exp_eps4.yaml` | `point_weighting.name: distance` |
| `bu_thr_right_001_exp_r3_continuous_eps4.yaml` | `bu_thr_right_001_exp_eps4.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128` |
| `bu_thr_right_01_exp_eps4.yaml` | `bu_no_pole_0001_exp_eps4.yaml` | `data.domain: [-3.9, -0.1]` |
| `bu_thr_right_01_exp_eps4_p3.yaml` | `bu_thr_right_01_exp_eps4.yaml` | `point_weighting.name: distance` |
| `bu_thr_right_01_exp_r3_continuous_eps4.yaml` | `bu_thr_right_01_exp_eps4.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128` |

Queues: `python -m pinn_feynman.training.queue configs/nonplanar_three_point/queues/<queue>.yaml`, or the notebook
`notebooks/run_local.ipynb`. A queue is idempotent: finished runs are skipped, interrupted ones resume.

| Queue | Runs | Configurations | Parts | Seeds |
|---|---|---|---|---|
| `queues/bu_thr.yaml` | 72 | `bu_thr_left_001_exp_eps4`, `bu_thr_left_001_exp_eps4_p3`, `bu_thr_left_001_exp_r3_continuous_eps4`, `bu_thr_left_01_exp_eps4`, `bu_thr_left_01_exp_eps4_p3`, `bu_thr_left_01_exp_r3_continuous_eps4`, `bu_thr_right_001_exp_eps4`, `bu_thr_right_001_exp_eps4_p3`, `bu_thr_right_001_exp_r3_continuous_eps4`, `bu_thr_right_01_exp_eps4`, `bu_thr_right_01_exp_eps4_p3`, `bu_thr_right_01_exp_r3_continuous_eps4` | imag, real | 0, 1, 2 |
