# configs/queues/

Training queues that span several problems: `python -m pinn_feynman.training.queue configs/queues/<queue>.yaml`, or the
notebook `notebooks/run_local.ipynb`. The configurations are given as `<problem>/<name>` (files in `configs/<problem>/`).

| Queue | Runs | Configurations | Parts | Seeds |
|---|---|---|---|---|
| `queues/bu_exp_replace.yaml` | 76 | `box_massless/bu_sched_exponential`, `box_massless/bu_sched_exponential_p3`, `box_one_mass/bu_sched_exponential`, `box_one_mass/bu_sched_exponential_epochs600`, `box_one_mass/bu_sched_exponential_lr3e-3`, `box_one_mass/bu_sched_exponential_p3`, `box_one_mass/bu_sched_exponential_points4x`, `box_one_mass/bu_sched_exponential_w128d3`, `box_one_mass/bu_sched_exponential_w128d3_points4x`, `sunrise_equal_mass/bu_phys_17_269_exp_p3`, `sunrise_equal_mass/bu_phys_2701_100_exp`, `sunrise_equal_mass/bu_phys_2701_100_exp_p3`, `sunrise_equal_mass/bu_phys_2701_100_exp_r3_continuous`, `sunrise_equal_mass/bu_phys_271_100_exp`, `sunrise_equal_mass/bu_phys_271_100_exp_p3`, `sunrise_equal_mass/bu_phys_271_100_exp_r3_continuous`, `sunrise_equal_mass/bu_phys_271_37_exp_p3`, `sunrise_equal_mass/bu_phys_301_2699_exp`, `sunrise_equal_mass/bu_phys_301_2699_exp_p3`, `sunrise_equal_mass/bu_phys_301_2699_exp_r3_continuous`, `sunrise_equal_mass/bu_phys_31_269_exp`, `sunrise_equal_mass/bu_phys_31_269_exp_p3`, `sunrise_equal_mass/bu_phys_31_269_exp_r3_continuous` | imag, real | 0, 1, 2 |
| `queues/bu_seeds_3x3.yaml` | 21 | `box_massless/bu_margin_01pc_exp`, `box_one_mass/bu_near_001`, `sunrise_equal_mass/bu_phys_2701_100`, `sunrise_equal_mass/bu_phys_271_100`, `sunrise_equal_mass/bu_phys_271_37_exp`, `sunrise_equal_mass/bu_phys_28_37_exp`, `sunrise_equal_mass/bu_phys_above_exp` | imag | 0, 1, 2 |
