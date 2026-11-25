"""Collocation points on fixed base points (Boni, thesis sec. 6.1.1, eqs. 6.7-6.9) or fresh each epoch.

`BasePointsSampler`: before training, N_base points are generated uniformly in the domain and each
gets a selection probability; at each epoch N_coll = n_per_epoch distinct base points are drawn
according to those probabilities and fed to the network in batches. Probabilities are `uniform`
(default) or one draw of a Dirichlet distribution (`dirichlet`, Boni's set-up; concentration 1 by default).

`ResampleSampler`: n_per_epoch new uniform points every epoch, no fixed points (no per-point state).

Both work on any domain of `data.domains`: an `Interval` (or a pair (lo, hi)) for the massless box, a
`Triangle` for the one-mass box; points have shape (N, d) (`Interval.sample_columns`, `Interval.sample_torch`).
"""

import numpy as np
import torch

from ..data.domains import as_domain

PROBABILITIES = ("uniform", "dirichlet")


def _check_sizes(n_total, n_per_epoch, batch_size):
    if n_per_epoch > n_total:
        raise ValueError("n_per_epoch cannot exceed n_base")
    if n_per_epoch % batch_size:
        raise ValueError("n_per_epoch must be a multiple of batch_size")


class BasePointsSampler:
    """Fixed base points with selection probabilities; `epoch_indices()` returns the batches of one epoch.

    Points are in the network variable (domain given in that variable), batches of shape (batch_size, d).
    With uniform probabilities and n_base == n_per_epoch every base point is used once per epoch.
    """

    fixed_points = True
    multi_variable = True            # works on a domain of any dimension

    def __init__(self, domain, n_base, n_per_epoch, batch_size, probabilities="uniform", concentration=None,
                 seed=None, dtype=torch.float64):
        _check_sizes(n_base, n_per_epoch, batch_size)
        if probabilities not in PROBABILITIES:
            raise ValueError(f"probabilities must be one of {PROBABILITIES}, got {probabilities!r}")
        if probabilities == "uniform" and concentration is not None:
            raise ValueError("concentration is a parameter of the Dirichlet probabilities, not of uniform ones")
        self.domain = as_domain(domain)
        self.batch_size, self.n_per_epoch, self.n_base = batch_size, n_per_epoch, n_base
        self.kind = probabilities
        rng = np.random.default_rng(seed)
        self.base_points = torch.as_tensor(self.domain.sample_columns(n_base, rng), dtype=dtype)
        if probabilities == "dirichlet":     # same random stream as the original Dirichlet sampler
            alpha = 1.0 if concentration is None else float(concentration)
            self.probabilities = torch.as_tensor(rng.dirichlet(np.full(n_base, alpha)))
        else:
            self.probabilities = torch.full((n_base,), 1.0 / n_base, dtype=torch.float64)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def to(self, device):
        """Move the base points to `device` (indices are always drawn on the CPU)."""
        self.base_points = self.base_points.to(device)
        return self

    def state_dict(self):
        """Random state of the per-epoch draws (base points and probabilities follow from the seed)."""
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])

    @property
    def batches_per_epoch(self):
        return self.n_per_epoch // self.batch_size

    def epoch_indices(self):
        """Index batches of a new epoch: n_per_epoch distinct base points, in random order.

        Indices let the caller reuse quantities precomputed on the base points (for
        example the connection matrix, as in Boni's collocation set, eq. 6.7).
        """
        if self.kind == "dirichlet":
            idx = torch.multinomial(self.probabilities, self.n_per_epoch, replacement=False, generator=self.generator)
        else:                                # same law as multinomial with equal probabilities, faster
            idx = torch.randperm(self.n_base, generator=self.generator)[:self.n_per_epoch]
        return list(torch.split(idx, self.batch_size))

    def epoch(self):
        """Point batches of a new epoch, shape (batch_size, d) each."""
        return [self.base_points[idx] for idx in self.epoch_indices()]


def DirichletCollocationSampler(domain, n_base, n_per_epoch, batch_size, concentration=1.0, seed=None,
                                dtype=torch.float64):
    """Boni's sampler with Dirichlet probabilities: a `BasePointsSampler`."""
    return BasePointsSampler(domain, n_base, n_per_epoch, batch_size, probabilities="dirichlet",
                             concentration=concentration, seed=seed, dtype=dtype)


class ResampleSampler:
    """n_per_epoch new uniform points every epoch (no per-point state).

    `base_points` holds the points of the current epoch and `version` counts the redraws, so the
    trainer knows when to recompute the quantities it precomputes on them (connection matrix).
    Points are drawn with the torch generator only: its state is the whole checkpoint.
    """

    fixed_points = False
    multi_variable = True            # works on a domain of any dimension

    def __init__(self, domain, n_per_epoch, batch_size, seed=None, dtype=torch.float64):
        _check_sizes(n_per_epoch, n_per_epoch, batch_size)
        self.domain = as_domain(domain)
        self.batch_size, self.n_per_epoch, self.dtype = batch_size, n_per_epoch, dtype
        self.device = torch.device("cpu")
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.version = 0
        self._draw()

    def _draw(self):
        points = self.domain.sample_torch(self.n_per_epoch, self.generator, dtype=self.dtype)
        self.base_points = points.to(device=self.device)
        self.version += 1

    def to(self, device):
        self.device = torch.device(device)
        self.base_points = self.base_points.to(self.device)
        return self

    def state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])

    @property
    def batches_per_epoch(self):
        return self.n_per_epoch // self.batch_size

    def epoch_indices(self):
        """Draw the points of a new epoch; index batches into them (already i.i.d., kept in order)."""
        self._draw()
        return list(torch.split(torch.arange(self.n_per_epoch), self.batch_size))

    def epoch(self):
        return [self.base_points[idx] for idx in self.epoch_indices()]
