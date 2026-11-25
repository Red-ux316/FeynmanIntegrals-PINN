# FeynmanIntegrals-PINN

Physics-informed neural networks (PINNs) for the differential equations of Feynman master integrals, in a bottom-up
formulation.

This is the code of the project for the QCD course of prof. P. Mastrolia (University of Padua). It extends the master's
thesis of G. Boni (*Differential Equations for Feynman Integrals and Physics Informed Neural Network*, Padua,
A.Y. 2023/2024, supervisor M. K. Mandal, co-supervisor P. Mastrolia).

## What the code does

The master integrals of a diagram satisfy a system of first-order differential equations in the kinematic variables,
order by order in the dimensional regulator ε. A network is trained on the boundary values and on the residual of the
equations at collocation points, the two losses balanced by ReLoBRaLo, as in Boni's thesis.

- **Bottom-up formulation.** The network learns only the masters of the top sector; the sub-topologies are known in
  closed form and enter its equation as a forcing term.
- **Four diagrams:** the one-loop massless box and one-mass box, the two-loop equal-mass sunrise for s > 3, and the
  non-planar two-loop three-point function with a massive loop.
- **True values:** closed formulas for the boxes and the non-planar function; for the sunrise, integration of the
  equation along the real axis from three starting points computed with AMFlow
  (`src/pinn_feynman/reference/starting_values.csv`).
- **Training methods:** Boni's baseline (uniform collocation points), R3 sampling on the continuous domain, and a
  distance weighting of the residual near the singular points; hyperparameter tuning with Optuna.

## Repository

| Folder | Content |
|---|---|
| [src/pinn_feynman/](src/pinn_feynman/README.md) | the package: systems of equations, true values, datasets, network, losses, training, evaluation, figures |
| [configs/](configs/README.md) | one folder per diagram: the training configurations (YAML) and the queues of trainings |
| [notebooks/](notebooks/README.md) | `run_local`: runs one or more queues on this machine, then evaluates the finished runs |
| [frobenius/](frobenius/README.md) | Frobenius expansion of the massless box around $x=0$: analytic and singular part, error of the truncated series |
| [docs/](docs/README.md) | tutorial and code map |

## Installation

Requirements: Anaconda or Miniconda, git.

```
git clone <repository-url> FeynmanIntegrals-PINN
cd FeynmanIntegrals-PINN
conda env create -f environment.yml
conda activate pinn-feynman
pip install -e . --no-deps
```

The environment uses the conda `defaults` channel and a CPU build of PyTorch (see `environment.yml`).

## Usage

Datasets and runs are written to `$PINN_FEYNMAN_STORAGE` if the variable is set, otherwise to the folder `storage/` at
the root of the repository (`storage/data/`: cached datasets; `storage/runs/`: one folder per training).

Build the datasets of a configuration (true values at the boundary, validation and test points; computed once, then
read from the cache):

```
python -m pinn_feynman.data.build configs/box_massless/bu_baseline.yaml
```

Train the network, real and imaginary part (about 6 s per epoch on one CPU thread for the massless box, so about
40 minutes for 400 epochs):

```
python -m pinn_feynman.training.run configs/box_massless/bu_baseline.yaml --part real
python -m pinn_feynman.training.run configs/box_massless/bu_baseline.yaml --part imag
```

An interrupted training resumes from its last checkpoint:

```
python -m pinn_feynman.training.run --resume storage/runs/<run>
```

Run a queue of trainings (finished runs are skipped, interrupted ones resume), or open `notebooks/run_local.ipynb`:

```
python -m pinn_feynman.training.queue configs/box_massless/queues/bottom_up.yaml
```

Evaluate a trained run on the test set (errors as in Boni's thesis, figures, comparison with the numbers of the thesis):

```
python -m pinn_feynman.evaluation.evaluate storage/runs/<real run> storage/runs/<imaginary run>
```

Step by step: [docs/tutorial.md](docs/tutorial.md).

## References

- G. Boni, *Differential Equations for Feynman Integrals and Physics Informed Neural Network*, master's thesis,
  University of Padua (2024).
- Z. Bern, L. Dixon, D. A. Kosower, *Dimensionally Regulated Pentagon Integrals*, hep-ph/9306240 (true values of
  the boxes).
- A. von Manteuffel, L. Tancredi, arXiv:1701.05905 (the non-planar three-point function).
- A. Daw et al., arXiv:2207.02338 (R3 sampling).
- R. Bischof, M. Kraus, arXiv:2110.09813 (ReLoBRaLo).

## License

MIT, see [LICENSE](LICENSE).
