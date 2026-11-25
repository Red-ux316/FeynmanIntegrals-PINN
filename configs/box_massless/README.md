# configs/box_massless/

Configurations of the one-loop massless box (Boni §6.2.1), bottom-up: the network learns the box M3 at the
orders $\varepsilon^{-2}\ldots\varepsilon^{2}$ (5 outputs), and the two bubbles M1, M2 enter its equation as the
forcing term. Variable $x=t/s$ in the physical region of the $s$ channel, $-1<x<0$ (singular points $x=0$ and
$x=-1$); the network input is $10x$ (`physics.input_scale: 10`). The domain is an interval
(`data.domain: [lo, hi]`).

Each file starts from its base (`base:`) and writes only what changes; the table lists those changes
(`python -m pinn_feynman.devtools.list_configs box_massless` prints the same tree).

| File | Base | What it sets (with respect to its base) |
|---|---|---|
| `baseline.yaml` | — | complete configuration |
| `bu_baseline.yaml` | `baseline.yaml` | `physics.system: box_massless_bu` |
| `bu_baseline_gradrec.yaml` | `bu_baseline.yaml` | `training.diagnostics.gradients.every: 64` |
| `bu_baseline_p3.yaml` | `bu_baseline.yaml` | `point_weighting.name: distance` |
| `bu_cdata100.yaml` | `bu_baseline.yaml` | `balancing.scales: [100, 1]` |
| `bu_exp_cdata100.yaml` | `bu_baseline.yaml` | `balancing.scales: [100, 1]`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97163` |
| `bu_long2000.yaml` | `bu_sched_exp_1000ep.yaml` | `training.epochs: 2000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.99426` |
| `bu_margin_01pc.yaml` | `bu_baseline.yaml` | `data.domain: [-0.9, -0.001]`; `data.datasets.test.n_nodes: 1024` |
| `bu_margin_01pc_exp.yaml` | `bu_margin_01pc.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97163`; `training.diagnostics.gradients.every: 64` |
| `bu_margin_01pc_exp_1000ep.yaml` | `bu_margin_01pc_exp.yaml` | `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.988553`; `training.diagnostics.gradients.every: 64` |
| `bu_margin_01pc_exp_fixed_w01.yaml` | `bu_margin_01pc_exp.yaml` | `balancing.name: fixed`; `balancing.weights: [0.1, 1.0]` |
| `bu_margin_01pc_exp_fixed_w1.yaml` | `bu_margin_01pc_exp.yaml` | `balancing.name: fixed`; `balancing.weights: [1.0, 1.0]` |
| `bu_margin_01pc_exp_fixed_w10.yaml` | `bu_margin_01pc_exp.yaml` | `balancing.name: fixed`; `balancing.weights: [10.0, 1.0]` |
| `bu_margin_01pc_exp_fixed_w100.yaml` | `bu_margin_01pc_exp.yaml` | `balancing.name: fixed`; `balancing.weights: [100.0, 1.0]` |
| `bu_margin_01pc_exp_fixed_w1000.yaml` | `bu_margin_01pc_exp.yaml` | `balancing.name: fixed`; `balancing.weights: [1000.0, 1.0]` |
| `bu_margin_01pc_exp_p3.yaml` | `bu_margin_01pc_exp.yaml` | `point_weighting.name: distance` |
| `bu_margin_01pc_r3_continuous_exp.yaml` | `bu_margin_01pc_exp.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_0001_00001.yaml` | `bu_sched_exponential.yaml` | `data.domain: [-0.001, -0.0001]`; `data.datasets.test.n_nodes: 1024`; `physics.input_scale: 10000`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_0001_00001_p3.yaml` | `bu_near_0001_00001.yaml` | `point_weighting.name: distance` |
| `bu_near_0001_00001_r3_continuous.yaml` | `bu_near_0001_00001.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_001_0001.yaml` | `bu_sched_exponential.yaml` | `data.domain: [-0.01, -0.001]`; `data.datasets.test.n_nodes: 1024`; `physics.input_scale: 1000`; `training.diagnostics.gradients.every: 64` |
| `bu_near_001_0001_p3.yaml` | `bu_near_001_0001.yaml` | `point_weighting.name: distance` |
| `bu_near_001_0001_r3_continuous.yaml` | `bu_near_001_0001.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_01_0001.yaml` | `bu_sched_exponential.yaml` | `data.domain: [-0.1, -0.001]`; `data.datasets.test.n_nodes: 1024`; `physics.input_scale: 100`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_01_0001_p3.yaml` | `bu_near_01_0001.yaml` | `point_weighting.name: distance` |
| `bu_near_01_0001_r3_continuous.yaml` | `bu_near_01_0001.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.gradients.every: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_03_0001.yaml` | `bu_sched_exponential.yaml` | `data.domain: [-0.3, -0.001]`; `data.datasets.test.n_nodes: 1024`; `training.diagnostics.gradients.every: 64` |
| `bu_near_03_0001_p3.yaml` | `bu_near_03_0001.yaml` | `point_weighting.name: distance` |
| `bu_near_03_0001_r3_continuous.yaml` | `bu_near_03_0001.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_m1.yaml` | `bu_sched_exponential.yaml` | `data.domain: [-0.999, -0.9]`; `data.datasets.test.n_nodes: 1024`; `training.diagnostics.gradients.every: 64` |
| `bu_near_m1_far.yaml` | `bu_sched_exponential.yaml` | `data.domain: [-0.99, -0.7]`; `data.datasets.test.n_nodes: 1024`; `training.diagnostics.gradients.every: 64` |
| `bu_near_m1_far_r3_continuous.yaml` | `bu_near_m1_far.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_near_m1_long.yaml` | `bu_near_m1.yaml` | `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.988553`; `training.diagnostics.gradients.every: 64` |
| `bu_near_m1_p3.yaml` | `bu_near_m1.yaml` | `point_weighting.name: distance` |
| `bu_near_m1_r3_continuous.yaml` | `bu_near_m1.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.diagnostics.snapshot_epochs: [1, 10, 50, 100, 200, 300, 400]` |
| `bu_r3_continuous_exp.yaml` | `bu_baseline.yaml` | `collocation.name: r3_continuous`; `collocation.n_per_epoch: 65536`; `collocation.batch_size: 64`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97163` |
| `bu_r3_continuous_exp_gradrec.yaml` | `bu_r3_continuous_exp.yaml` | `training.diagnostics.gradients.every: 64` |
| `bu_sched_exp_1000ep.yaml` | `bu_sched_exponential.yaml` | `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.988553`; `training.diagnostics.gradients.every: 64` |
| `bu_sched_exponential.yaml` | `bu_baseline.yaml` | `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97163` |
| `bu_sched_exponential_gradrec.yaml` | `bu_sched_exponential.yaml` | `training.diagnostics.gradients.every: 64` |
| `bu_sched_exponential_p3.yaml` | `bu_sched_exponential.yaml` | `point_weighting.name: distance` |
| `bu_w64d4_exp.yaml` | `bu_baseline.yaml` | `model.hidden: [64, 64, 64, 64]`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.97163` |
| `bu_w64d4_exp_1000ep.yaml` | `bu_w64d4_exp.yaml` | `training.epochs: 1000`; `training.scheduler.name: exponential`; `training.scheduler.gamma: 0.988553`; `training.diagnostics.gradients.every: 64` |
| `bu_w64d4_exp_gradrec.yaml` | `bu_w64d4_exp.yaml` | `training.diagnostics.gradients.every: 64` |

Queues: `python -m pinn_feynman.training.queue configs/box_massless/queues/<queue>.yaml`, or the notebook
`notebooks/run_local.ipynb`. A queue is idempotent: finished runs are skipped, interrupted ones resume.

| Queue | Runs | Configurations | Parts | Seeds |
|---|---|---|---|---|
| `queues/bottom_up.yaml` | 6 | `bu_baseline` | imag, real | 0, 1, 2 |
| `queues/bu_explore.yaml` | 6 | `bu_cdata100`, `bu_exp_cdata100`, `bu_margin_01pc`, `bu_r3_continuous_exp`, `bu_sched_exponential`, `bu_w64d4_exp` | real | 0 |
| `queues/bu_explore2.yaml` | 5 | `bu_baseline_gradrec`, `bu_margin_01pc_exp`, `bu_sched_exp_1000ep` | real | 0, 1, 2 |
| `queues/bu_fixed_w_grid.yaml` | 18 | `bu_margin_01pc_exp`, `bu_margin_01pc_exp_fixed_w01`, `bu_margin_01pc_exp_fixed_w1`, `bu_margin_01pc_exp_fixed_w10`, `bu_margin_01pc_exp_fixed_w100`, `bu_margin_01pc_exp_fixed_w1000` | real | 0, 1, 2 |
| `queues/bu_gradrec.yaml` | 13 | `bu_baseline_gradrec`, `bu_r3_continuous_exp_gradrec`, `bu_sched_exp_1000ep`, `bu_sched_exponential_gradrec`, `bu_w64d4_exp_gradrec` | imag, real | 0, 1, 2, 3, 4 |
| `queues/bu_imag.yaml` | 6 | `bu_baseline`, `bu_sched_exp_1000ep` | imag | 0, 1, 2 |
| `queues/bu_long2000.yaml` | 6 | `bu_long2000`, `bu_sched_exp_1000ep` | real | 0, 1, 2 |
| `queues/bu_m1_far.yaml` | 12 | `bu_near_m1_far`, `bu_near_m1_far_r3_continuous` | imag, real | 0, 1, 2 |
| `queues/bu_margin_long.yaml` | 6 | `bu_margin_01pc_exp`, `bu_margin_01pc_exp_1000ep` | real | 0, 1, 2 |
| `queues/bu_near_m1_long.yaml` | 6 | `bu_near_m1`, `bu_near_m1_long` | real | 0, 1, 2 |
| `queues/bu_near_more.yaml` | 12 | `bu_near_0001_00001`, `bu_near_0001_00001_r3_continuous`, `bu_near_01_0001`, `bu_near_01_0001_r3_continuous` | real | 0, 1, 2 |
| `queues/bu_near_singularity.yaml` | 9 | `bu_near_001_0001`, `bu_near_03_0001`, `bu_near_m1` | real | 0, 1, 2 |
| `queues/bu_p3.yaml` | 48 | `bu_baseline`, `bu_baseline_p3`, `bu_margin_01pc_exp`, `bu_margin_01pc_exp_p3`, `bu_near_0001_00001`, `bu_near_0001_00001_p3`, `bu_near_001_0001`, `bu_near_001_0001_p3`, `bu_near_01_0001`, `bu_near_01_0001_p3`, `bu_near_03_0001`, `bu_near_03_0001_p3`, `bu_near_m1`, `bu_near_m1_p3` | imag, real | 0, 1, 2 |
| `queues/bu_r3_all.yaml` | 6 | `bu_near_03_0001_r3_continuous`, `bu_near_m1_r3_continuous` | real | 0, 1, 2 |
| `queues/bu_r3_near.yaml` | 6 | `bu_margin_01pc_r3_continuous_exp`, `bu_near_001_0001_r3_continuous` | real | 0, 1, 2 |
| `queues/bu_seeds.yaml` | 18 | `bu_baseline`, `bu_margin_01pc_exp`, `bu_r3_continuous_exp`, `bu_sched_exp_1000ep`, `bu_sched_exponential`, `bu_w64d4_exp` | real | 0, 1, 2 |
| `queues/bu_seeds2.yaml` | 15 | `bu_baseline`, `bu_sched_exp_1000ep`, `bu_sched_exponential` | real | 0, 1, 2, 3, 4 |
| `queues/bu_w64d4_exp_1000ep_seeds.yaml` | 6 | `bu_baseline`, `bu_w64d4_exp_1000ep` | real | 0, 1, 2 |
