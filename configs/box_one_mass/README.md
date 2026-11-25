# configs/box_one_mass/

Configurations of the one-loop one-mass box (Boni §6.2.2), bottom-up: the network learns the box
(1 master × 5 orders), and the three bubbles enter its equation as the forcing term. Variables $(s_1, s_2) = (t, m^2)$
with $s = 1$; the domain is the triangle $s_1<0$, $s_2>0$, $1+s_1-s_2>0$, with a margin $\delta$ from its sides
(`data.domain: {name: triangle, margin: δ}`).

Each file starts from its base (`base:`) and writes only what changes; the table lists those changes
(`python -m pinn_feynman.devtools.list_configs box_one_mass` prints the same tree).

| File | Base | What it sets (with respect to its base) |
|---|---|---|
| `baseline.yaml` | — | complete configuration |
| `bu_baseline.yaml` | `baseline.yaml` | `physics.system: box_one_mass_bu` |
| `bu_baseline_p3.yaml` | `bu_baseline.yaml` | `point_weighting.name: distance` |
| `bu_epochs600.yaml` | `bu_baseline.yaml` | `training.epochs: 600` |
| `bu_exp_cdata100.yaml` | `bu_baseline.yaml` | `balancing.scales: [100, 1]`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97724` |
| `bu_lr3e-3.yaml` | `bu_baseline.yaml` | `training.optimizer.lr: 0.003` |
| `bu_near_0001.yaml` | `bu_sched_exponential.yaml` | `data.domain.name: triangle`; `data.domain.margin: 0.001` |
| `bu_near_0001_fixed_w01.yaml` | `bu_near_0001.yaml` | `balancing.name: fixed`; `balancing.weights: [0.1, 1.0]` |
| `bu_near_0001_fixed_w1.yaml` | `bu_near_0001.yaml` | `balancing.name: fixed`; `balancing.weights: [1.0, 1.0]` |
| `bu_near_0001_fixed_w10.yaml` | `bu_near_0001.yaml` | `balancing.name: fixed`; `balancing.weights: [10.0, 1.0]` |
| `bu_near_0001_fixed_w100.yaml` | `bu_near_0001.yaml` | `balancing.name: fixed`; `balancing.weights: [100.0, 1.0]` |
| `bu_near_0001_fixed_w1000.yaml` | `bu_near_0001.yaml` | `balancing.name: fixed`; `balancing.weights: [1000.0, 1.0]` |
| `bu_near_0001_p3.yaml` | `bu_near_0001.yaml` | `point_weighting.name: distance` |
| `bu_near_0001_r3_continuous.yaml` | `bu_near_0001.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128` |
| `bu_near_001.yaml` | `bu_sched_exponential.yaml` | `data.domain.name: triangle`; `data.domain.margin: 0.01` |
| `bu_near_001_p3.yaml` | `bu_near_001.yaml` | `point_weighting.name: distance` |
| `bu_near_001_r3_continuous.yaml` | `bu_near_001.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128` |
| `bu_points4x.yaml` | `bu_baseline.yaml` | `collocation.n_per_epoch: 262144` |
| `bu_r3_continuous_exp.yaml` | `bu_baseline.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 128`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97724` |
| `bu_sched_exponential.yaml` | `bu_baseline.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97724` |
| `bu_sched_exponential_epochs600.yaml` | `bu_sched_exponential.yaml` | `training.epochs: 600`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.988553` |
| `bu_sched_exponential_lr3e-3.yaml` | `bu_sched_exponential.yaml` | `training.optimizer.lr: 0.003`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.973665` |
| `bu_sched_exponential_p3.yaml` | `bu_sched_exponential.yaml` | `point_weighting.name: distance` |
| `bu_sched_exponential_points4x.yaml` | `bu_sched_exponential.yaml` | `collocation.n_per_epoch: 262144` |
| `bu_sched_exponential_w128d3.yaml` | `bu_sched_exponential.yaml` | `model.hidden: [128, 128, 128]` |
| `bu_sched_exponential_w128d3_points4x.yaml` | `bu_sched_exponential.yaml` | `model.hidden: [128, 128, 128]`; `collocation.n_per_epoch: 262144` |
| `bu_w128d3.yaml` | `bu_baseline.yaml` | `model.hidden: [128, 128, 128]` |
| `bu_w128d3_points4x.yaml` | `bu_w128d3.yaml` | `collocation.n_per_epoch: 262144` |

Queues: `python -m pinn_feynman.training.queue configs/box_one_mass/queues/<queue>.yaml`, or the notebook
`notebooks/run_local.ipynb`. A queue is idempotent: finished runs are skipped, interrupted ones resume.

| Queue | Runs | Configurations | Parts | Seeds |
|---|---|---|---|---|
| `queues/bottom_up.yaml` | 8 | `bu_baseline` | imag, real | 0, 1, 2, 3 |
| `queues/bu_explore.yaml` | 3 | `bu_exp_cdata100`, `bu_r3_continuous_exp`, `bu_sched_exponential` | real | 0 |
| `queues/bu_explore2.yaml` | 4 | `bu_epochs600`, `bu_lr3e-3`, `bu_points4x`, `bu_w128d3` | real | 0 |
| `queues/bu_fixed_w_grid.yaml` | 18 | `bu_near_0001`, `bu_near_0001_fixed_w01`, `bu_near_0001_fixed_w1`, `bu_near_0001_fixed_w10`, `bu_near_0001_fixed_w100`, `bu_near_0001_fixed_w1000` | real | 0, 1, 2 |
| `queues/bu_gradrec.yaml` | 4 | `bu_points4x` | imag | 0, 1, 2, 3 |
| `queues/bu_imag.yaml` | 8 | `bu_baseline`, `bu_points4x` | imag | 0, 1, 2, 3 |
| `queues/bu_near.yaml` | 12 | `bu_near_0001`, `bu_near_0001_r3_continuous`, `bu_near_001`, `bu_near_001_r3_continuous` | real | 0, 1, 2 |
| `queues/bu_p3.yaml` | 24 | `bu_baseline`, `bu_baseline_p3`, `bu_near_0001`, `bu_near_0001_p3`, `bu_near_001`, `bu_near_001_p3` | imag, real | 0, 1, 2 |
| `queues/bu_r3_seeds.yaml` | 2 | `bu_sched_exponential` | real | 1, 2 |
| `queues/bu_seeds.yaml` | 20 | `bu_baseline`, `bu_lr3e-3`, `bu_points4x`, `bu_r3_continuous_exp`, `bu_w128d3` | real | 0, 1, 2, 3 |
| `queues/bu_w128d3_points4x_seeds.yaml` | 6 | `bu_baseline`, `bu_w128d3_points4x` | real | 0, 1, 2 |
