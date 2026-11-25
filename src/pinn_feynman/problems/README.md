# pinn_feynman.problems

The diagrams the pipeline can train on: **one file per problem**. The file declares the problem and nothing else; the registry (`_registry.py`) only finds it by name.

| File | What it does |
|---|---|
| `_registry.py` | `Problem` (frozen dataclass: `name`, `system`, `reference`, `boni`, `parts`, and for bottom-up `base`, `unknowns`, `provider`), `get_problem(name)`, `available_problems()`, `ProblemMapping` |
| `box_massless.py` | `PROBLEM` of the one-loop massless box (one variable, closed form) |
| `box_one_mass.py` | `PROBLEM` of the one-loop one-mass box (two variables, series in Nielsen polylogarithms) |
| `box_massless_bu.py`, `box_one_mass_bu.py` | bottom-up variants: the network learns only the box, the bubbles come from a provider |
| `sunrise_equal_mass_positive.py` | two-loop equal-mass sunrise, $m^2=3$, for $s>0$: domains between the singular points 0, 3, 27, values by integration of (A.10) along the real axis from AMFlow points; real and imaginary part (the imaginary part is zero below 27, where only the real part is trained) |
| `sunrise_equal_mass_positive_bu.py` | bottom-up sunrise: the network learns the coupled block $(I_{11100},I_{21100})$, the tadpole $I_{11000}$ comes from a constant provider |
| `nonplanar_three_point.py` | non-planar two-loop three-point function (one variable $x=-s/m^2$, both parts): the two top masters with the closed-form forcing of the subtopologies. A data problem that already declares `unknowns=(0, 1)` and a forcing `provider` (without it its equation would be the homogeneous one) |
| `nonplanar_three_point_bu.py` | the same problem with `base="nonplanar_three_point"`, for the `*_bu` naming of configurations and queues |

**Parts:** `parts` (default `('real', 'imag')`) lists the parts of the true values that can be trained. A problem whose masters are real in the whole domain declares `('real',)`: `training.setup.build_experiment` then refuses `part: imag` with an error instead of training a network on zeros. A bottom-up problem may only declare parts of its base.

- `Problem` holds dotted names, resolved on first use: `system` (module of `pinn_feynman.systems`), `reference` (module of `pinn_feynman.reference` with `reference_values`, `ORDERS`, `CACHE_PARAMS`, `COMPUTE_OPTIONS`), `boni` (`"module:NAME"`, optional). Importing a problem file costs a tiny import; sympy and mpmath come in only when the system or the reference is used.
- `get_problem(name)` imports `problems/<name>.py` and checks that it defines `PROBLEM` with `name` equal to the file name. `available_problems()` lists the files that do not start with an underscore. An unknown name is a `ValueError` listing the available ones.
- `ProblemMapping` is the read-only dictionary view behind `physics.SYSTEMS`, `data.generation.GENERATORS`/`COMPUTE_OPTIONS` and `evaluation.boni_reference.BONI`.

**Bottom-up problems:** `unknowns` (columns of the basis the network learns) and `provider` (`'module:factory'`, light constructor) go together; `base` names the problem whose data, cache, reference and Boni results are used (same `system` and `reference` strings), so `data.problem` stays the base and no data is regenerated while `physics.system` is the `*_bu` name. `data_problem()`, `columns()`, `build_provider()`. A bottom-up problem is not in `data.generation.GENERATORS`.

**Adding a diagram:** write `problems/<name>.py`, `systems/<name>.py` and `reference/<name>.py` (contract in `docs/tutorial.md`, section 6), then the configurations. No existing file is edited.

