"""Training loop of Boni's PINN (thesis sec. 4.4, 6.1.2, Table 6.1).

Each optimisation step uses the boundary data (eq. 6.11) and one batch of collocation
points (eq. 6.12); the two losses are combined by the balancer (eq. 6.13). After each
epoch the validation loss (eq. 5.16) is computed and the best model is kept (sec. 5, p. 81).
The whole state can be saved and restored, so an interrupted training resumes exactly.

With `residual_stats_points > 0`, every epoch also records the propagation statistics of the
residual (mean, max, skewness, kurtosis; Daw et al. App. H) on a fixed uniform grid: no gradient,
no random numbers, so the training itself is unchanged. One variable: a uniform grid between the ends of the base
points (fixed points) or of the domain. More variables: the domain's fixed lattice (`Triangle.grid`: distinct
points, vertices and edges included), built without random numbers.

With a `recorder` (training/recorder.py, configuration `training.diagnostics`) every step and epoch are also
observed: effective points, unweighted physics loss, share of the loss near x = 0, gradients of the loss terms,
snapshots of the state.

With `validation_physics` (default on) every epoch also records `val_physics_loss`, the unweighted mean squared
residual (L_PDE) at the validation points; their connection matrices are computed once. No gradient, no random numbers.

Samplers without fixed points (`resample`, `r3_continuous`) change their points; their `version` tells the trainer when
to recompute the connection matrix: after the draw of an epoch, right after each R3 refresh and after a resume. A
sampler with `blocks_per_epoch` > 1 (`r3_continuous` with a small population) is refreshed between the blocks of an
epoch too; the end-of-epoch refresh is the last one.
"""

import copy
import time
from dataclasses import dataclass

import torch

from .. import progress as console
from ..physics.derivatives import value_and_derivative
from ..physics.losses import cascade_residual, data_loss, residual_norm, weighted_physics_loss
from ..physics.propagation import propagation_stats
from ..weighting import NoWeighting
from .history import History


@dataclass
class PointSet:
    """Inputs z (N, d) in the network variables (d = 1: (N, 1)) and targets y (N, n_masters, n_orders)."""

    z: torch.Tensor
    y: torch.Tensor

    def to(self, device):
        return PointSet(self.z.to(device), self.y.to(device))


class Trainer:
    def __init__(self, model, system, boundary, validation, sampler, balancer, optimizer, scheduler,
                 log_every=0, device="cpu", point_weighting=None, residual_stats_points=0, recorder=None,
                 validation_physics=False):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.boundary = boundary.to(self.device)
        self.validation = validation.to(self.device)
        self.sampler = sampler.to(self.device)
        self.balancer = balancer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every = log_every
        self.point_weighting = point_weighting if point_weighting is not None else NoWeighting()
        if hasattr(self.point_weighting, "to"):
            self.point_weighting.to(self.device)
        # connection matrix on all base points, computed once (Boni's collocation set, eq. 6.7);
        # recomputed when a sampler without fixed points draws new ones (its `version` changes)
        self.system = system
        self._update_matrices()
        # fixed uniform grid over the range of the base points (the domain for fresh points),
        # for the per-epoch residual statistics
        self.stats_grid = None
        if residual_stats_points and getattr(getattr(self.sampler, "domain", None), "dim", 1) > 1:
            base = self.sampler.base_points
            self.stats_grid = torch.as_tensor(self.sampler.domain.grid(residual_stats_points),
                                              dtype=base.dtype, device=self.device)
            self.coefficients_grid = system.coefficients(self.stats_grid)
        elif residual_stats_points:          # one variable: the original grid, operation for operation
            base = self.sampler.base_points
            lo, hi = ((base.min().item(), base.max().item()) if getattr(self.sampler, "fixed_points", True)
                      else self.sampler.domain)
            self.stats_grid = torch.linspace(lo, hi, residual_stats_points,
                                             dtype=base.dtype, device=self.device).reshape(-1, 1)
            self.coefficients_grid = system.coefficients(self.stats_grid)
        # connection matrix at the validation points, for the validation physics loss (computed once)
        self.validation_physics = bool(validation_physics)
        self._unweighted_sum, self._unweighted_steps = 0.0, 0     # plain L_PDE of the epoch with per-point weights
        if self.validation_physics:
            self.coefficients_val = system.coefficients(self.validation.z)
        # optional training recorder (`training.diagnostics`): observes, never changes the training
        self.recorder = recorder.bind(self) if recorder is not None else None
        self.history = History()
        self.epoch = 0
        self.best_state = None
        self.best_val = float("inf")
        self.best_epoch = None

    def _update_matrices(self, force=False):
        """Connection matrix on the sampler's current points, if they changed since the last call (or `force`)."""
        version = getattr(self.sampler, "version", None)
        if force or not hasattr(self, "coefficients_base") or version != self._matrices_version:
            self.coefficients_base = self.system.coefficients(self.sampler.base_points)
            self._matrices_version = version

    # A_0 and A_1 of the cached coefficients (the names of the complete system; a bottom-up system adds a forcing)
    @property
    def a0_base(self):
        return self.coefficients_base.orders[0]

    @property
    def a1_base(self):
        return self.coefficients_base.orders[1]

    @property
    def a0_grid(self):
        return self.coefficients_grid.orders[0]

    @property
    def a1_grid(self):
        return self.coefficients_grid.orders[1]

    @property
    def a0_val(self):
        return self.coefficients_val.orders[0]

    @property
    def a1_val(self):
        return self.coefficients_val.orders[1]

    # --- one step, one epoch --------------------------------------------------------------
    def residual(self, idx):
        """Cascade residual (6.5) at the base points idx, shape (B, n_masters, n_orders)."""
        y, dy = value_and_derivative(self.model, self.sampler.base_points[idx])
        c = self.coefficients_base[idx]
        return cascade_residual(y, dy, *c.orders, forcing=c.forcing)

    @torch.no_grad()
    def residual_norms(self, idx, chunk=16384):
        """Per-point residual norms at many base points, without gradients (for R3)."""
        idx = idx.to(self.device)
        return torch.cat([residual_norm(self.residual(part)) for part in torch.split(idx, chunk)])

    def losses(self, idx):
        """[L_data, L_DE] for the collocation batch with base-point indices idx.

        L_DE is Boni's mean squared residual, or <(w r)^2> with the per-point weights w of
        the point weighting (e.g. distance weighting) times those of the sampler, if it has any
        (`loss_weights(idx)`).
        """
        idx = idx.to(self.device)
        l_data = data_loss(self.model(self.boundary.z), self.boundary.y)
        residual = self.residual(idx)
        z = self.sampler.base_points[idx]
        norms = residual_norm(residual).detach()
        weights = self.point_weighting(idx, z, norms)
        importance = self.sampler.loss_weights(idx) if hasattr(self.sampler, "loss_weights") else None
        if importance is not None:           # the sampler's own weights
            weights = importance if weights is None else weights * importance
        if self.recorder is not None:
            self.recorder.on_batch(idx, z, norms, weights)
        if weights is not None and getattr(self.point_weighting, "log_unweighted", False):
            # the plain L_PDE of the batch, for `physics_loss_unweighted` (weightings with `log_unweighted`, e.g.
            # distance weighting; the recorder, when on, writes the same column)
            self._unweighted_sum += float(torch.mean(norms**2))
            self._unweighted_steps += 1
        return [l_data, weighted_physics_loss(residual, weights)]

    def train_step(self, idx):
        self.optimizer.zero_grad(set_to_none=True)
        terms = self.losses(idx)
        total = self.balancer(terms)
        if self.recorder is not None and self.recorder.wants_gradients():
            self.recorder.on_gradients(self, terms)          # observes only (autograd.grad, graph kept)
        total.backward()
        self.optimizer.step()
        return total.item(), terms[0].item(), terms[1].item()

    def train_epoch(self):
        """Mean weighted loss, mean L_data, mean L_DE over the steps of one epoch.

        A sampler with `blocks_per_epoch` > 1 is refreshed between its blocks (not after the last one: that refresh
        is done by `_epoch`, as for the samplers refreshed once per epoch).
        """
        sums, steps = [0.0, 0.0, 0.0], 0
        self._unweighted_sum, self._unweighted_steps = 0.0, 0
        for block in range(getattr(self.sampler, "blocks_per_epoch", 1)):
            if block:
                self._refresh_sampler()
            batches = self.sampler.epoch_indices()
            self._update_matrices()
            for idx in batches:
                for k, v in enumerate(self.train_step(idx)):
                    sums[k] += v
            steps += len(batches)
        return [s / steps for s in sums]

    def _refresh_sampler(self):
        """R3 update of the population with the current network, then the matrices of the new points (if new)."""
        self.sampler.refresh(self.residual_norms)
        self._update_matrices()

    @torch.no_grad()
    def residual_stats(self):
        """Propagation statistics of the residual norms on the fixed grid: {residual_mean, _max, _skewness, _kurtosis}."""
        y, dy = value_and_derivative(self.model, self.stats_grid)
        c = self.coefficients_grid
        norms = residual_norm(cascade_residual(y, dy, *c.orders, forcing=c.forcing))
        return {f"residual_{k}": v for k, v in propagation_stats(norms.cpu().numpy()).items()}

    @torch.no_grad()
    def validate(self):
        """Eq. (5.16): MSE between the network and the true values at the validation points."""
        return data_loss(self.model(self.validation.z), self.validation.y).item()

    @torch.no_grad()
    def validate_physics(self):
        """Unweighted physics loss mean(r^2) (Boni's L_DE, eq. 6.12) at the validation points."""
        y, dy = value_and_derivative(self.model, self.validation.z)
        c = self.coefficients_val
        return torch.mean(cascade_residual(y, dy, *c.orders, forcing=c.forcing) ** 2).item()

    # --- training loop -----------------------------------------------------------------------
    def fit(self, epochs, on_epoch_end=None, desc="train", leave=True):
        """Train until `epochs` epochs in total (continues from self.epoch after a resume).

        `on_epoch_end(trainer)` is called after each epoch (e.g. to write checkpoints). A progress bar
        shows the epochs with the last losses; every `log_every` epochs a line is left above it.
        """
        with console.bar(epochs, desc, unit="ep", initial=self.epoch, leave=leave) as bar:
            while self.epoch < epochs:
                row = self._epoch()
                bar.set_postfix_str(f"train {row['train_loss']:.2e}  val {row['val_loss']:.2e}  "
                                    f"best {self.best_val:.2e}  lr {row['lr']:.0e}", refresh=False)
                bar.update(1)
                if self.log_every and self.epoch % self.log_every == 0:
                    console.note(f"  epoch {self.epoch:4d}  train {row['train_loss']:.3e}  data {row['data_loss']:.3e}"
                                 f"  DE {row['physics_loss']:.3e}  val {row['val_loss']:.3e}  lr {row['lr']:.1e}"
                                 f"  ({row['seconds']:.1f} s)")
                if on_epoch_end is not None:
                    on_epoch_end(self)
        return self.history

    def _epoch(self):
        """One epoch: training steps, end-of-epoch updates, validation, history row."""
        self.epoch += 1
        start = time.perf_counter()
        if hasattr(self.balancer, "set_epoch"):             # balancers that depend on the epoch
            self.balancer.set_epoch(self.epoch)
        train, l_data, l_de = self.train_epoch()
        self.point_weighting.end_epoch()
        trained_population = getattr(self.sampler, "population", None)    # R3 refresh replaces it
        trained_points = (self.sampler.population_points()                 # coordinates, for the recorder only
                          if self.recorder is not None and hasattr(self.sampler, "population_points") else None)
        if hasattr(self.sampler, "refresh"):
            self._refresh_sampler()
        val = self.validate()
        val_physics = self.validate_physics() if self.validation_physics else None
        lr = self.optimizer.param_groups[0]["lr"]
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train)        # the plateau scheduler needs the metric
        else:
            self.scheduler.step()             # cosine, exponential: a function of the epoch only
        if val < self.best_val:
            self.best_val, self.best_epoch = val, self.epoch
            self.best_state = copy.deepcopy(self.model.state_dict())
        row = {"epoch": self.epoch, "train_loss": train, "data_loss": l_data, "physics_loss": l_de,
               "val_loss": val, "lr": lr, "seconds": time.perf_counter() - start}
        if val_physics is not None:
            row["val_physics_loss"] = val_physics
        if self._unweighted_steps:            # weightings with `log_unweighted`: the plain L_PDE too
            row["physics_loss_unweighted"] = self._unweighted_sum / self._unweighted_steps
        weights = getattr(self.balancer, "weights", None)
        if weights is not None:
            row.update({f"weight_{i}": float(w) for i, w in enumerate(weights)})
        row.update(self._method_diagnostics())
        if self.stats_grid is not None:
            row.update(self.residual_stats())
        if self.recorder is not None:
            row.update(self.recorder.end_epoch(self, trained_population, trained_points))
        self.history.append(row)
        return row

    def _method_diagnostics(self):
        """Extra per-epoch columns of the methods: the point weighting's own (`history_columns()`), retained points."""
        out = {}
        if hasattr(self.point_weighting, "history_columns"):
            out.update(self.point_weighting.history_columns())
        if hasattr(self.sampler, "last_retained"):
            out["r3_retained"] = int(self.sampler.last_retained)
        return out

    # --- checkpoint ----------------------------------------------------------------------------
    def state_dict(self):
        state = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "balancer": self.balancer.state_dict(),
            "sampler": self.sampler.state_dict(),
            "point_weighting": self.point_weighting.state_dict(),
            "history": self.history.state_dict(),
            "best_state": self.best_state,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
        }
        if self.recorder is not None:
            state["recorder"] = self.recorder.state_dict()
        return state

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.balancer.load_state_dict(state["balancer"])
        self.sampler.load_state_dict(state["sampler"])
        if not getattr(self.sampler, "fixed_points", True):
            self._update_matrices(force=True)         # new coordinates, maybe with the same version number
        self.point_weighting.load_state_dict(state.get("point_weighting", {}))
        self.history.load_state_dict(state["history"])
        self.best_state, self.best_val, self.best_epoch = state["best_state"], state["best_val"], state["best_epoch"]
        if self.recorder is not None:
            self.recorder.load_state_dict(state.get("recorder"))
