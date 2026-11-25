"""R3 sampling on the continuous domain (Daw et al., arXiv:2207.02338, Algorithm 1).

`R3ContinuousSampler`: a population of points is trained on for one pass; then the residual norm F(x) = |R(x)| is
evaluated on the whole population, the points with F above the mean are retained and the others are replaced by
points drawn uniformly from the continuous domain, as in Algorithm 1. Optionally a smaller population, refreshed
several times per epoch.
"""

import torch

from ..data.domains import as_domain


class R3ContinuousSampler:
    """R3 on the continuous domain: the population is a tensor of coordinates, no base points.

    - `population` (default n_per_epoch) points form the population; `epoch_indices()` returns one
      pass over it, shuffled, in batches of indices into `base_points` (= the population itself).
    - `refresh(residual_fn)`: F = residual_fn(arange(population)) on the whole population (the
      trainer's matrices still belong to the population just trained on), points with F > mean(F)
      are retained (first in the new tensor), the others replaced by uniform draws on the domain.
      The new population is a NEW tensor and `version` grows by one: the trainer recomputes the
      connection matrix on it (as for `resample`).
    - With population == n_per_epoch the population is refreshed once per epoch (by the trainer, at
      the end of the epoch). With a smaller population the epoch is split into
      `blocks_per_epoch` = n_per_epoch / population blocks; each block is one pass over the population,
      i.e. population / batch_size steps, and the trainer refreshes between blocks and at the end of
      the epoch. Every point therefore gets exactly one gradient step between two refreshes.

    Random numbers come from the torch generator only (shuffles and new points): its state, the
    population, `version` and `last_retained` are the whole checkpoint.
    """

    fixed_points = False
    multi_variable = True            # works on a domain of any dimension (Interval, Triangle)

    def __init__(self, domain, n_per_epoch, batch_size, population=None, seed=None, dtype=torch.float64):
        size = n_per_epoch if population is None else int(population)
        if size < batch_size or size % batch_size:
            raise ValueError("population must be a positive multiple of batch_size")
        if n_per_epoch % size:
            raise ValueError("n_per_epoch must be a multiple of population")
        self.domain = as_domain(domain)
        self.batch_size, self.n_per_epoch, self.size, self.dtype = batch_size, n_per_epoch, size, dtype
        self.device = torch.device("cpu")
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.version = 0
        self.points = self._uniform(size)
        self.version += 1
        self.last_retained = 0

    def _uniform(self, n):
        return self.domain.sample_torch(n, self.generator, self.dtype).to(device=self.device)

    @property
    def base_points(self):
        """The population: the trainer indexes it as it indexes the base points of the other samplers."""
        return self.points

    @property
    def batches_per_epoch(self):
        return self.n_per_epoch // self.batch_size

    @property
    def blocks_per_epoch(self):
        """Passes over the population per epoch; the trainer refreshes after each (1 = once per epoch)."""
        return self.n_per_epoch // self.size

    def to(self, device):
        self.device = torch.device(device)
        self.points = self.points.to(self.device)
        return self

    def epoch_indices(self):
        """One pass over the population (one block), shuffled, in batches of indices into `base_points`."""
        order = torch.randperm(self.size, generator=self.generator)
        return list(torch.split(order, self.batch_size))

    def epoch(self):
        return [self.base_points[idx] for idx in self.epoch_indices()]

    def population_points(self):
        """Coordinates (population, d) of the current population, network variable."""
        return self.points

    @property
    def refreshes(self):
        """Refreshes done so far (the population of the constructor is version 1)."""
        return self.version - 1

    # three hooks of `refresh`, for subclasses that change the rule; here they are Algorithm 1
    def _skip_refresh(self):
        """True when this call must change nothing."""
        return False

    def _cutoff(self, f):
        """Points with F above this value are retained."""
        return f.mean()

    def _replace(self, keep):
        """The new population, retained points first."""
        retained = self.points[keep]
        return torch.cat([retained, self._uniform(self.size - len(retained))])

    def refresh(self, residual_fn):
        """One R3 update on the continuum: retain F > mean(F), draw the rest uniformly on the domain."""
        if self._skip_refresh():
            return self.last_retained
        f = residual_fn(torch.arange(self.size)).cpu()
        keep = (f > self._cutoff(f)).to(self.points.device)
        self.last_retained = int(keep.sum())
        self.points = self._replace(keep)
        self.version += 1
        return self.last_retained

    def state_dict(self):
        return {"generator": self.generator.get_state(), "points": self.points.detach().cpu().clone(),
                "version": self.version, "last_retained": self.last_retained}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self.points = state["points"].to(device=self.device, dtype=self.dtype).clone()
        self.version = state["version"]
        self.last_retained = state["last_retained"]
