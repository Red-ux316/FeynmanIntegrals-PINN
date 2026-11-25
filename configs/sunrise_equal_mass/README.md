# configs/sunrise_equal_mass/

Configurations of the two-loop equal-mass sunrise (Boni §6.3.2), $m^2=3$, for $s>3$
(problem `sunrise_equal_mass_positive`), bottom-up: the network learns the coupled block $(I_{11100}, I_{21100})$ at the
orders $\varepsilon^{-2}\ldots\varepsilon^0$, and the tadpole $I_{11000}$ enters their equations in closed form as
the forcing term. The masters are real below $s=27$ and complex above it (singular points $s=3$ and $s=27$); the
network input is $s$ itself (`physics.input_scale: 1`). The domain is an interval (`data.domain: [lo, hi]`). Every
configuration starts from `bu_base.yaml`, which holds the settings shared by all of them.

Each file starts from its base (`base:`) and writes only what changes; the table lists those changes
(`python -m pinn_feynman.devtools.list_configs sunrise_equal_mass` prints the same tree).

| File | Base | What it sets (with respect to its base) |
|---|---|---|
| `bu_base.yaml` | — | complete configuration |
| `bu_phys_17_269.yaml` | `bu_phys_3_27.yaml` | `data.domain: [17.0, 26.9]`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_17_269_exp.yaml` | `bu_phys_17_269.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_17_269_exp_p3.yaml` | `bu_phys_17_269_exp.yaml` | `point_weighting.name: distance` |
| `bu_phys_17_269_exp_r3_continuous.yaml` | `bu_phys_17_269_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_17_269_p3.yaml` | `bu_phys_17_269.yaml` | `point_weighting.name: distance` |
| `bu_phys_17_269_r3_continuous.yaml` | `bu_phys_17_269.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_17_269_w128d4_exp_1000ep.yaml` | `bu_phys_17_269.yaml` | `model.hidden: [128, 128, 128, 128]`; `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.993116`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_2701_100.yaml` | `bu_phys_above.yaml` | `data.domain: [27.01, 100.0]` |
| `bu_phys_2701_100_exp.yaml` | `bu_phys_2701_100.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_2701_100_exp_p3.yaml` | `bu_phys_2701_100_exp.yaml` | `point_weighting.name: distance` |
| `bu_phys_2701_100_exp_r3_continuous.yaml` | `bu_phys_2701_100_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_2701_100_p3.yaml` | `bu_phys_2701_100.yaml` | `point_weighting.name: distance` |
| `bu_phys_2701_100_r3_continuous.yaml` | `bu_phys_2701_100.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_100.yaml` | `bu_phys_above.yaml` | `data.domain: [27.1, 100.0]` |
| `bu_phys_271_100_exp.yaml` | `bu_phys_271_100.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_100_exp_p3.yaml` | `bu_phys_271_100_exp.yaml` | `point_weighting.name: distance` |
| `bu_phys_271_100_exp_r3_continuous.yaml` | `bu_phys_271_100_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_100_p3.yaml` | `bu_phys_271_100.yaml` | `point_weighting.name: distance` |
| `bu_phys_271_100_r3_continuous.yaml` | `bu_phys_271_100.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_37.yaml` | `bu_phys_above.yaml` | `data.domain: [27.1, 37.0]`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_271_37_exp.yaml` | `bu_phys_271_37.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_37_exp_p3.yaml` | `bu_phys_271_37_exp.yaml` | `point_weighting.name: distance` |
| `bu_phys_271_37_exp_r3_continuous.yaml` | `bu_phys_271_37_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_37_p3.yaml` | `bu_phys_271_37.yaml` | `point_weighting.name: distance` |
| `bu_phys_271_37_r3_continuous.yaml` | `bu_phys_271_37.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_271_37_w128d4_exp_1000ep.yaml` | `bu_phys_271_37.yaml` | `model.hidden: [128, 128, 128, 128]`; `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.993116`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_28_37.yaml` | `bu_phys_above.yaml` | `data.domain: [28.0, 37.0]`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_28_37_exp.yaml` | `bu_phys_28_37.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_28_37_exp_r3_continuous.yaml` | `bu_phys_28_37_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_28_37_r3_continuous.yaml` | `bu_phys_28_37.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_301_2699.yaml` | `bu_phys_3_27.yaml` | `data.domain: [3.01, 26.99]` |
| `bu_phys_301_2699_exp.yaml` | `bu_phys_301_2699.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_301_2699_exp_p3.yaml` | `bu_phys_301_2699_exp.yaml` | `point_weighting.name: distance` |
| `bu_phys_301_2699_exp_r3_continuous.yaml` | `bu_phys_301_2699_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_301_2699_p3.yaml` | `bu_phys_301_2699.yaml` | `point_weighting.name: distance` |
| `bu_phys_301_2699_r3_continuous.yaml` | `bu_phys_301_2699.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_31_269.yaml` | `bu_phys_3_27.yaml` | `data.domain: [3.1, 26.9]` |
| `bu_phys_31_269_exp.yaml` | `bu_phys_31_269.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_31_269_exp_p3.yaml` | `bu_phys_31_269_exp.yaml` | `point_weighting.name: distance` |
| `bu_phys_31_269_exp_r3_continuous.yaml` | `bu_phys_31_269_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_31_269_p3.yaml` | `bu_phys_31_269.yaml` | `point_weighting.name: distance` |
| `bu_phys_31_269_r3_continuous.yaml` | `bu_phys_31_269.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_3_27.yaml` | `bu_base.yaml` | `data.problem: sunrise_equal_mass_positive`; `data.domain: [4.0, 24.0]`; `physics.system: sunrise_equal_mass_positive_bu`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_3_27_exp.yaml` | `bu_phys_3_27.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_3_27_exp_r3_continuous.yaml` | `bu_phys_3_27_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_3_27_r3_continuous.yaml` | `bu_phys_3_27.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_3_27_w128d4_exp_1000ep.yaml` | `bu_phys_3_27.yaml` | `model.hidden: [128, 128, 128, 128]`; `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.993116`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_above.yaml` | `bu_base.yaml` | `data.problem: sunrise_equal_mass_positive`; `data.domain: [30.0, 100.0]`; `physics.system: sunrise_equal_mass_positive_bu`; `training.diagnostics.gradients.every: 64` |
| `bu_phys_above_exp.yaml` | `bu_phys_above.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.98288`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_above_exp_r3_continuous.yaml` | `bu_phys_above_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_phys_above_r3_continuous.yaml` | `bu_phys_above.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |

Queues: `python -m pinn_feynman.training.queue configs/sunrise_equal_mass/queues/<queue>.yaml`, or the notebook
`notebooks/run_local.ipynb`. A queue is idempotent: finished runs are skipped, interrupted ones resume.

| Queue | Runs | Configurations | Parts | Seeds |
|---|---|---|---|---|
| `queues/bu_near_singularity.yaml` | 9 | `bu_phys_17_269`, `bu_phys_271_37` | imag, real | 0, 1, 2 |
| `queues/bu_p3.yaml` | 42 | `bu_phys_17_269`, `bu_phys_17_269_p3`, `bu_phys_2701_100`, `bu_phys_2701_100_p3`, `bu_phys_271_100`, `bu_phys_271_100_p3`, `bu_phys_271_37`, `bu_phys_271_37_p3`, `bu_phys_301_2699`, `bu_phys_301_2699_p3`, `bu_phys_31_269`, `bu_phys_31_269_p3` | imag, real | 0, 1, 2 |
| `queues/bu_phys_3_27_w128d4_exp_1000ep.yaml` | 6 | `bu_phys_3_27`, `bu_phys_3_27_w128d4_exp_1000ep` | real | 0, 1, 2 |
| `queues/bu_phys_near.yaml` | 24 | `bu_phys_2701_100`, `bu_phys_2701_100_r3_continuous`, `bu_phys_271_100`, `bu_phys_271_100_r3_continuous`, `bu_phys_301_2699`, `bu_phys_301_2699_r3_continuous`, `bu_phys_31_269`, `bu_phys_31_269_r3_continuous` | real | 0, 1, 2 |
| `queues/bu_physical.yaml` | 3 | `bu_phys_3_27`, `bu_phys_above` | imag, real | 0 |
| `queues/bu_physical2.yaml` | 6 | `bu_phys_28_37` | imag, real | 0, 1, 2 |
| `queues/bu_physical_seeds.yaml` | 3 | `bu_phys_3_27` | real | 0, 1, 2 |
| `queues/bu_r3_all.yaml` | 5 | `bu_phys_17_269_r3_continuous`, `bu_phys_271_37_r3_continuous`, `bu_phys_28_37_r3_continuous`, `bu_phys_3_27_r3_continuous`, `bu_phys_above_r3_continuous` | real | 0 |
| `queues/bu_r3_exp.yaml` | 30 | `bu_phys_17_269_exp`, `bu_phys_17_269_exp_r3_continuous`, `bu_phys_271_37_exp`, `bu_phys_271_37_exp_r3_continuous`, `bu_phys_28_37_exp`, `bu_phys_28_37_exp_r3_continuous`, `bu_phys_3_27_exp`, `bu_phys_3_27_exp_r3_continuous`, `bu_phys_above_exp`, `bu_phys_above_exp_r3_continuous` | real | 0, 1, 2 |
| `queues/bu_threshold_w128d4_exp_1000ep.yaml` | 12 | `bu_phys_17_269`, `bu_phys_17_269_w128d4_exp_1000ep`, `bu_phys_271_37`, `bu_phys_271_37_w128d4_exp_1000ep` | real | 0, 1, 2 |
