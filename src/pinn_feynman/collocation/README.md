# pinn_feynman.collocation

Collocation points for the physics loss (Boni §6.1.1, eq. 6.7–6.9; Tab. 6.1 "Sampling").

Domains: every sampler accepts a `data.domains.Domain` (`Interval`, `Triangle`) or a `(lo, hi)` pair (`as_domain`) and returns points of shape `(N, d)` (`d = 1` for the massless box, 2 for the one-mass box); on the triangle the points are uniform by reflection. The class attribute `multi_variable = True` marks the samplers that work on a domain of any dimension (all three here); `training.setup.check_multi_variable` refuses the others when the system has more than one variable.

Every sampler has `fixed_points`: `True` when the points are fixed for the whole training (base points), `False` when they are drawn anew. Methods that keep a state per point need fixed points: `training.setup.check_collocation` refuses the other combinations.

## `sampler.py`

### `BasePointsSampler(domain, n_base, n_per_epoch, batch_size, probabilities="uniform", concentration=None, seed, dtype)`
1. **At the start:** `n_base` uniform base points in the domain, in the network's variable. Each gets a selection probability:
   - `probabilities: uniform` (default): all equal;
   - `probabilities: dirichlet`: **one** draw from a Dirichlet distribution (6.8) with all parameters equal to `concentration` (default 1). Large concentration → probabilities close to $1/N_{base}$; small → very different.
2. **At each epoch** (`epoch_indices()`): `n_per_epoch` distinct base points drawn according to the probabilities, in random order, split into batches. With `uniform` the draw is a random permutation (same law, faster); with `n_base == n_per_epoch` every point is used exactly once per epoch.

The base points depend only on the seed, so `uniform` and `dirichlet` with the same seed have the same points; the per-epoch draws differ.

Boni's values for the box: $N_{base}=2\cdot10^5$, $N_{coll}$ = 65,536 points per epoch, batches of 64, hence 1024 batches (Tab. 6.1: "Base points size", "Samples per partition", "Batch size", "Partition size").

`DirichletCollocationSampler(domain, n_base, n_per_epoch, batch_size, concentration=1.0, seed, dtype)`: Boni's sampler, a `BasePointsSampler` with Dirichlet probabilities.

### `ResampleSampler(domain, n_per_epoch, batch_size, seed, dtype)`
`n_per_epoch` **new** uniform points every epoch, drawn with the sampler's torch generator only (its state is the whole checkpoint). This is the sampling of the baseline. `base_points` holds the points of the current epoch and `version` counts the draws: the trainer recomputes the connection matrix when it changes. No per-point state is possible.

| Name | What it does |
|---|---|
| `base_points` | tensor `(n_base, d)` (fresh sampler: `(n_per_epoch, d)`, current epoch) |
| `probabilities` | tensor `(n_base,)` (base points only) |
| `batches_per_epoch` | `n_per_epoch // batch_size` |
| `epoch_indices()` | batches of **indices** into `base_points` for a new epoch (to reuse precomputed quantities, such as the matrix) |
| `epoch()` | list of the point batches for a new epoch |
| `to(device)` | moves the points |
| `state_dict()`, `load_state_dict()` | state of the random generator (checkpoint) |

## `r3.py` — `R3ContinuousSampler(domain, n_per_epoch, batch_size, population=None, seed, dtype)`

**R3** (retain-resample-release, Daw et al., arXiv:2207.02338, Algorithm 1) on the continuous domain. The population is a tensor of coordinates and the replaced points are drawn **uniformly from the continuous domain**, as in Algorithm 1.

1. `population` points (default `n_per_epoch`) form the population; `base_points` is the population itself, so the trainer indexes it as for the other samplers;
2. `epoch_indices()` returns **one pass** over the population, shuffled, in batches of indices;
3. `refresh(residual_fn)`: residual norms $F$ on the whole population (`residual_fn(arange(population))`, with the matrices of the population just trained on); the points with $F$ above the mean are retained (first in the new tensor), the others replaced by uniform draws. The result is a **new** tensor and `version` grows by one, so the trainer recomputes the connection matrix (as for `resample`).

Two regimes, from the configuration:

| Configuration | Population | Refresh |
|---|---|---|
| `{name: r3_continuous, n_per_epoch: 65536, batch_size: 64}` | 65,536 | once per epoch (after 1024 steps) |
| `{name: r3_continuous, n_per_epoch: 65536, batch_size: 64, population: 1024}` | 1,024 | every pass over the population: `population / batch_size` = 16 steps, 64 times per epoch |

In both, every point gets exactly one gradient step between two refreshes, and an epoch has the same `n_per_epoch` points and batches as the baseline. `blocks_per_epoch` = `n_per_epoch / population`; the trainer refreshes between blocks and at the end of the epoch. Constraints: `population` a multiple of `batch_size`, `n_per_epoch` a multiple of `population`.

Difference from Algorithm 1: the paper takes a full-batch step on the whole population at every iteration; here the step uses a mini-batch of 64 and the refresh waits for a full pass over the population. The small population is the regime closest to the paper at a sustainable cost.

Random numbers: torch generator only (shuffles and new points). State: `{generator, points, version, last_retained}`; the trainer recomputes the matrices after loading it. `population_points()` returns the coordinates (network variable). No per-point state is possible.

## `factory.py`
`COLLOCATION_SAMPLERS` (registry: `base_points`, `resample`, `dirichlet` = `base_points` with `probabilities: dirichlet`, `r3_continuous`), `sampler_class(config)`, `build_collocation_sampler(config, domain, seed, dtype)` from the `collocation` section of the YAML.

**Checks**:
- points within the domain; positive probabilities summing to 1;
- 64 batches of 64 without repetitions;
- reproducibility, and epochs differing from each other;
- effect of the concentration;
- uniform distribution across multiple epochs (Kolmogorov–Smirnov test);
- the `dirichlet` sampler against saved reference points;
- uniform probabilities: same points as Dirichlet, every point once per epoch when `n_base == n_per_epoch`, nearly equal visit counts;
- fresh points: new and uniform every epoch, deterministic resume.

R3 checks: retained = F > mean, new tensor and version at every refresh, no ceiling near x = 0 (residual 1/|x|), exact resume in the middle of a run after refreshes (both regimes), matrices of the current population after every refresh and after a resume.
