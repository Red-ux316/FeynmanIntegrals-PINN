# pinn_feynman.data

Dataset of true values: boundary (training), validation and test points (Boni, eq. 6.6 and 6.10). Values are computed with `pinn_feynman.reference` and saved **once** in an on-disk cache, because they cost about 0.4 s per point.

## Modules

| Module | Name | What it does |
|---|---|---|
| `domains.py` | `Interval(lo, hi)`, `Triangle(margin, scale=1)` | the domain of a problem, one object for every problem: `dim`, `sample(n, seed)`, `boundary()`, `contains`, `scaled(c)`, `as_config()`. `Interval` is a `tuple` subclass (same hash and same JSON as the pair `(lo, hi)`: its cache key depends only on the end points). `Triangle` is the physical region of the one-mass box, sampled uniformly by reflection ($u+v>1\to(1-u,1-v)$, no point discarded); its vertices and `contains` come from `systems.box_one_mass`. Sampling for the training: `sample_columns(n, rng)` (numpy, shape `(N, d)`) and `sample_torch(n, generator, dtype)`; for `Interval` they are `rng.uniform(lo, hi, size=(n, 1))` and `torch.rand` then `lo + (hi - lo) * u`; `Triangle.grid(n)` is the fixed barycentric lattice used for the residual statistics (990 distinct points for `n=1024`, vertices and edges included, no random numbers) |
| | `as_domain(value)`, `DOMAINS` | a `Domain`, a pair `[lo, hi]` or a dict `{name: triangle, margin: …}` (from a YAML `data.domain`) → `Domain` |
| `sampling.py` | `boundary_points(domain)` | the boundary points of the domain: 2 endpoints (Boni §6.2.1: *«two points at the boundary»*), the 3 vertices of the triangle |
| | `uniform_points(domain, n, seed)` | `n` uniform points, reproducible from the seed |
| | `chebyshev_points(domain, n)` | Chebyshev nodes (second kind), for interpolation; only for one-dimensional domains |
| | `SAMPLERS`, `sample(name, domain, n, seed)` | registry of samplers, by name |
| `generation.py` | `GENERATORS`, `COMPUTE_OPTIONS` | views of `pinn_feynman.problems`: problem → (true-value generator, parameters that identify it) and → how the points are split (declared by `reference/<problem>.py`) |
| | `compute_reference(problem, xs, workers, chunk_size, progress)` | complex values `(N, N_masters, 5)`; with `workers > 1` uses multiple processes, in chunks that depend on the problem (`COMPUTE_OPTIONS`: 10 points for the massless box; one block of 10 000 points and one process for the one-mass box, whose fast method is vectorised). The result does not depend on `workers` or `chunk_size`. `progress`: `False`, `True`, or the label of the progress bar |
| `interpolation.py` | `chebyshev_nodes(domain, n)`, `barycentric_interpolate(nodes, values, xs)` | Chebyshev interpolation (barycentric formula, also for complex values) |
| `dataset.py` | `DatasetSpec` | what to generate: problem, sampler, domain (list = interval, dict = triangle), `n`, seed, `method` (`exact` or `chebyshev`), `n_nodes`. `from_config` reads a YAML entry |
| | `ReferenceDataset` | `x` `(N,)` for one variable, `(N, d)` for `d > 1`, and complex `values` `(N, N_masters, 5)`; `part("real"/"imag")`, `tensors(part)` → `(x (N,d), y (N,N_masters,5))` torch `float64` |
| `cache.py` | `cache_key(spec)`, `cache_path(spec)` | file name: hash of the spec **and** the generator's parameters, so a change to the generator invalidates the cache |
| | `save_dataset`, `load_dataset` | `.npz` file with `x`, `values` and JSON metadata |
| | `build_dataset(spec)` | samples and computes. With `method: chebyshev` it takes from the cache (or generates) the exact dataset at the nodes, `nodes_spec(spec)`, and interpolates (`interpolate_reference`) |
| | `get_dataset(spec)` | loads from the cache, or generates and saves |
| `check_interpolation.py` | `python -m pinn_feynman.data.check_interpolation <config.yaml>` | compares the test interpolant with the exact validation values |
| `build.py` | `python -m pinn_feynman.data.build <config.yaml> [more configs ...] [--workers N] [--only name ...]` | generates the datasets of the `data` sections: one line per dataset (`cached` or `computing`), a progress bar for each one computed |
| | `unique_specs(config_paths, only)`, `build_all(config_paths, workers, only)` | a dataset shared by several configurations is listed and built once |

## Interpolated test set
Computing exactly $10^5$ points would cost tens of hours on the shared CPU. The test set is therefore obtained from the exact values at 128 Chebyshev nodes, by interpolation. The coefficients are analytic on $[-0.9,-0.1]$, because the singularities lie at $x=0$ and $x=-1$, and the error decreases as $2^{-n}$. Accuracy is **measured** with `check_interpolation`, on the 1000 exact validation points.

## Where the cache lives
In `data/` at the repository root (not versioned), derived from the package location (`pinn_feynman.paths.data_dir`). Changed with the `PINN_FEYNMAN_DATA_DIR` environment variable.

The saved variable is $x=t/s$. The real and imaginary parts are Boni's two separate trainings (§6.2.1.1).

