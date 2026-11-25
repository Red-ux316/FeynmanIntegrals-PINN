"""Training recorder: what happens during the training, observed without changing it.

Configuration block `training.diagnostics` (absent = off; `{}` = on with the defaults):

    diagnostics:
      snapshot_epochs: [1, 10, 50, 100, 200, 400]   # full snapshots; the final epoch is always added
      log_cells: 24                                  # cells of equal width in log10|x| over the domain
      grid_points: 2048                              # fixed quadrature grid, uniform in log10|x|
      gradients: {every: 64}                         # optional: gradient of each loss term every 64 steps
      population_gradient: {stride: 8}               # optional: gradients on the whole epoch's points

Per batch (from the quantities the trainer already has: residual norms r_i, weights w_i or None; the step only
keeps references, the sums are computed vectorised at the end of the epoch):
c_i = w_i^2 r_i^2 (c_i = r_i^2 without weights), the batch's effective points n_eff = (sum c)^2 / sum c^2 weighted
and unweighted, the unweighted physics loss mean(r_i^2), and per log|x| cell sum r^2, sum c and the number of batch
points; with fixed base points also one visit counter per base point.

Per epoch, columns of history.csv:
- `physics_loss_unweighted`: mean over the batches of mean(r_i^2) (the plain loss, also with point weights);
- `n_eff_weighted`, `n_eff_unweighted`: mean over the batches of n_eff (batch size = all points equal, 1 = one point);
- `tail_share_r2`: share of the integral of r^2 over the domain in the tail x in (-0.1, 0), on the quadrature grid
  with the network at the end of the epoch (independent of how the points are sampled);
- `tail_share_w2r2`: share of the sum of c over the epoch's batches in the tail: where the loss the optimiser saw
  lived (depends on the sampling, e.g. R3 enriches the tail).
The two tail columns are left out when the domain has no tail. `diagnostics/cells.npz` keeps the per-cell sums of
every epoch, and the grid integral of r^2 per cell.

Snapshots `diagnostics/epoch_XXXX.npz` (end of the epoch, after the R3 refresh): network parameters, residual on
the grid, visit counters; the point weighting's own state, if it has `snapshot_arrays()`; R3: population after the refresh, population trained on in the epoch, retained points. The
population is stored as coordinates in the physical variable x, float64 (`population_x`, `population_trained_x`; the
first `last_retained` entries of `population_x` are the retained points), for every R3 sampler; the discrete R3 also
keeps the base-point indices (`population`, `population_trained`, int32). With a population refreshed several times
per epoch (`r3_continuous` with a small population) "trained on in the epoch" means the last block of the epoch.

Gradients per loss term (`gradients`, off by default): every `every` steps (the first step of each window),
before the optimiser step, g_k = grad of (coefficient_k * L_k) for k = data, PDE, where coefficient_k = lam_k c_k is
the factor of the term in the total (ReLoBRaLo weight times its constant scale). With `torch.autograd.grad` on the
graph of the step (kept with retain_graph; the trainer's `.grad` is untouched, no random numbers): |g_data|,
|g_pde|, cos(g_data, g_pde), and the same norms after Adam's division, |g / (sqrt(v_hat) + eps)|, with the
optimiser's second moment v before this step (bias-corrected; NaN before the first step): about the step each term
alone would cause. `diagnostics/gradients.npz` (one row per sampled step: epoch, step, coefficients, norms) and
per epoch the medians `grad_norm_data`, `grad_norm_pde`, `grad_cosine`, `adam_step_data`, `adam_step_pde` in
history.csv.

Gradients on the whole population (`population_gradient`, off by default): at the end of every epoch, with the
network of that moment (no step in between), the gradient of the plain L_PDE on the points trained on in the epoch
(P_t, every `stride`-th of them, in the order of the steps), of the L_PDE on the population after the R3 refresh
(P_{t+1}, samplers with a population; every `stride`-th point) and of L_data: the mean push of the equation, which
the per-step gradients above (mostly minibatch noise) do not show, and how much the refresh alone turns it.
History columns `pop_grad_norm_pde`, `pop_grad_norm_data`, `pop_grad_cos_data_pde`, `pop_grad_cos_refresh`
(cos(g_PDE(P_t), g_PDE(P_{t+1})); NaN without a population); the full vectors (float32, one row per epoch) in
`diagnostics/population_gradients.npz`, for later projections. Snapshots also keep Adam's moments (`adam.exp_avg`,
`adam.exp_avg_sq`, flattened in the order of the parameters, `adam.step`) and the ReLoBRaLo state
(`balancer.weights`, `balancer.previous`, `balancer.initial`).

Guarantees: the recorder never calls the point weighting (it would advance a weighting with a state), draws no
random numbers and works without gradients on detached tensors, so the training is bit-for-bit the same with it on
or off (test). Snapshots wait in memory and are written by `flush` (in `run.save_checkpoint`, before the
checkpoint): a snapshot lost to an interruption is recomputed identically after the resume.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch

from ..physics.derivatives import value_and_derivative
from ..physics.losses import cascade_residual, data_loss, physics_loss, residual_norm

TAIL = (-0.1, 0.0)                    # the tail next to the singular point x = 0: x in (-0.1, 0)
DEFAULTS = {"snapshot_epochs": [1, 10, 50, 100, 200, 400], "log_cells": 24, "grid_points": 2048, "gradients": None,
            "population_gradient": None}
DIRECTORY = "diagnostics"
CELLS_FILE = "cells.npz"
GRADIENTS_FILE = "gradients.npz"
POPULATION_GRADIENTS_FILE = "population_gradients.npz"
POPULATION_KEYS = ("g_data", "g_pde", "g_pde_next")
GRADIENT_KEYS = ("epoch", "step", "coef_data", "coef_pde", "norm_data", "norm_pde", "cosine", "adam_data", "adam_pde")


def snapshot_name(epoch):
    return f"epoch_{epoch:04d}.npz"


def log_abs_range(domain):
    """(min |x|, max |x|) over a domain that does not contain x = 0."""
    lo, hi = (float(v) for v in domain)
    if lo <= 0.0 <= hi:
        raise ValueError(f"training.diagnostics needs a domain without x = 0 (cells in log|x|), got {domain}")
    return min(abs(lo), abs(hi)), max(abs(lo), abs(hi))


def log_quadrature_grid(domain, n):
    """Nodes uniform in log10|x| over the domain (ascending x) and trapezoid weights in x: sum(weights) = hi - lo."""
    if n < 2:
        raise ValueError("grid_points must be at least 2")
    a, b = log_abs_range(domain)
    ax = np.logspace(math.log10(a), math.log10(b), n)
    ax[0], ax[-1] = a, b
    x = np.sort(math.copysign(1.0, float(domain[0])) * ax)
    weights = np.zeros(n)
    step = np.diff(x)
    weights[:-1] += step / 2
    weights[1:] += step / 2
    return x, weights


def effective_points(c):
    """n_eff = (sum c)^2 / sum c^2 of one batch (tensor); NaN when every c is zero."""
    return c.sum() ** 2 / (c * c).sum()


def _atomic_npz(path, arrays, compressed=False):
    """np.savez to a temporary file, then rename (an interruption never leaves a broken file)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as stream:
        (np.savez_compressed if compressed else np.savez)(stream, **arrays)
    os.replace(tmp, path)


class TrainingRecorder:
    """Per-batch and per-epoch diagnostics of a training; see the module docstring."""

    def __init__(self, domain, input_scale, epochs, snapshot_epochs=None, log_cells=None, grid_points=None,
                 gradients=None, population_gradient=None):
        self.domain = tuple(float(v) for v in domain)                   # physical variable x
        self.scale = float(input_scale)
        self.final_epoch = int(epochs)
        epochs_list = DEFAULTS["snapshot_epochs"] if snapshot_epochs is None else snapshot_epochs
        if any(int(e) < 1 for e in epochs_list):
            raise ValueError("snapshot_epochs must be positive epochs")
        self.snapshot_epochs = sorted({int(e) for e in epochs_list} | {self.final_epoch})
        self.n_cells = int(DEFAULTS["log_cells"] if log_cells is None else log_cells)
        if self.n_cells < 1:
            raise ValueError("log_cells must be at least 1")
        a, b = log_abs_range(self.domain)
        self.log_edges = np.linspace(math.log10(a), math.log10(b), self.n_cells + 1)
        self.grid_x, self.grid_weights = log_quadrature_grid(
            self.domain, int(DEFAULTS["grid_points"] if grid_points is None else grid_points))
        self.has_tail = self.domain[1] > TAIL[0] and self.domain[0] < TAIL[1]
        self.grid_tail = (self.grid_x > TAIL[0]) & (self.grid_x < TAIL[1])
        self.grid_cell = self._cells_np(self.grid_x)
        self.cells = {key: [] for key in ("epoch", "sum_r2", "sum_c", "count", "grid_int_r2")}
        self.pending = {}
        self.visits = None
        self.device = torch.device("cpu")
        if gradients is not None and (not isinstance(gradients, dict) or set(gradients) - {"every"}):
            raise ValueError("training.diagnostics.gradients must be {every: <steps>}")
        self.gradient_every = int(gradients.get("every", 64)) if gradients is not None else 0
        if gradients is not None and self.gradient_every < 1:
            raise ValueError("gradients.every must be at least 1")
        self.steps = 0                                  # optimisation steps seen (all epochs)
        self.gradients = {key: [] for key in GRADIENT_KEYS}
        self._epoch_gradient_start = 0
        if population_gradient is not None and (not isinstance(population_gradient, dict)
                                                or set(population_gradient) - {"stride"}):
            raise ValueError("training.diagnostics.population_gradient must be {stride: <points>}")
        self.population_stride = int(population_gradient.get("stride", 8)) if population_gradient is not None else 0
        if population_gradient is not None and self.population_stride < 1:
            raise ValueError("population_gradient.stride must be at least 1")
        self.population = {"epoch": [], **{key: [] for key in POPULATION_KEYS}}

    # --- set-up -------------------------------------------------------------------------------------
    def bind(self, trainer):
        """Grid matrices on the trainer's device; visit counters if the sampler has fixed base points."""
        base = trainer.sampler.base_points
        self.device = base.device
        self.grid_z = torch.as_tensor(self.scale * self.grid_x, dtype=base.dtype, device=self.device).reshape(-1, 1)
        self.coefficients_grid = trainer.system.coefficients(self.grid_z)
        if getattr(trainer.sampler, "fixed_points", True):
            self.visits = torch.zeros(len(base), dtype=torch.int32)     # 0.8 MB in the checkpoint for 200000 points
        self._reset_epoch()
        return self

    def _cells_np(self, x):
        cell = np.floor((np.log10(np.abs(x)) - self.log_edges[0]) / (self.log_edges[1] - self.log_edges[0]))
        return np.clip(cell.astype(np.int64), 0, self.n_cells - 1)

    def _reset_epoch(self):
        self._batches = []                    # (idx, z, norms, weights) of every step, references only

    # --- per batch ----------------------------------------------------------------------------------
    def on_batch(self, idx, z, norms, weights):
        """Keep one optimisation step: base-point indices, inputs z (B, 1), residual norms (B,), weights or None.

        Only references are stored (about a microsecond per step); everything is computed in `end_epoch`.
        """
        self._batches.append((idx.detach(), z.detach(), norms.detach(),
                              None if weights is None else weights.detach()))
        self.steps += 1

    def wants_gradients(self):
        """True at the first step of each window of `every` steps (after `on_batch` of that step)."""
        return self.gradient_every > 0 and (self.steps - 1) % self.gradient_every == 0

    def on_gradients(self, trainer, terms):
        """Gradient of each weighted term at this step (graph kept for the trainer's backward); see the docstring."""
        params = [p for p in trainer.model.parameters() if p.requires_grad]
        balancer = trainer.balancer
        coefs = balancer.coefficients() if hasattr(balancer, "coefficients") else balancer.weights
        coefs = [float(c) for c in coefs]
        flat = []
        for coef, term in zip(coefs, terms):
            grads = torch.autograd.grad(coef * term, params, retain_graph=True, allow_unused=True)
            flat.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                                   for p, g in zip(params, grads)]).to(torch.float64))
        g_data, g_pde = flat
        n_data, n_pde = float(g_data.norm()), float(g_pde.norm())
        cosine = float(g_data @ g_pde / (n_data * n_pde)) if n_data > 0 and n_pde > 0 else float("nan")
        adam = self._adam_denominator(trainer.optimizer, params)
        a_data = float((g_data / adam).norm()) if adam is not None else float("nan")
        a_pde = float((g_pde / adam).norm()) if adam is not None else float("nan")
        for key, value in zip(GRADIENT_KEYS, (trainer.epoch, self.steps, coefs[0], coefs[1], n_data, n_pde,
                                              cosine, a_data, a_pde)):
            self.gradients[key].append(value)

    def _population_gradients(self, trainer):
        """Gradients of L_PDE on P_t and P_{t+1} and of L_data with the current network (see the docstring)."""
        params = [p for p in trainer.model.parameters() if p.requires_grad]

        def flat(loss):
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                              for p, g in zip(params, grads)]).to(torch.float64)

        def cos(a, b):
            na, nb = float(a.norm()), float(b.norm())
            return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

        with torch.enable_grad():
            z_t = torch.cat([z for _, z, _, _ in self._batches])[::self.population_stride]
            g_pde = flat(physics_loss(trainer.model, trainer.system, z_t))
            g_data = flat(data_loss(trainer.model(trainer.boundary.z), trainer.boundary.y))
            g_next = None
            if hasattr(trainer.sampler, "population_points"):
                z_next = trainer.sampler.population_points()[::self.population_stride]
                g_next = flat(physics_loss(trainer.model, trainer.system, z_next))
        self.population["epoch"].append(trainer.epoch)
        self.population["g_data"].append(g_data.float().cpu())
        self.population["g_pde"].append(g_pde.float().cpu())
        self.population["g_pde_next"].append((torch.full_like(g_pde, float("nan")) if g_next is None
                                               else g_next).float().cpu())
        return {"pop_grad_norm_pde": float(g_pde.norm()), "pop_grad_norm_data": float(g_data.norm()),
                "pop_grad_cos_data_pde": cos(g_data, g_pde),
                "pop_grad_cos_refresh": cos(g_pde, g_next) if g_next is not None else float("nan")}

    @staticmethod
    @torch.no_grad()
    def _adam_denominator(optimizer, params):
        """sqrt(v_hat) + eps of Adam/AdamW, flattened in the order of `params`; None before the first step."""
        state_of = {}
        for group in optimizer.param_groups:
            beta2, eps = group.get("betas", (None, None))[1], group.get("eps")
            for p in group["params"]:
                state_of[id(p)] = (optimizer.state.get(p, {}), beta2, eps)
        parts = []
        for p in params:
            state, beta2, eps = state_of.get(id(p), ({}, None, None))
            if beta2 is None or "exp_avg_sq" not in state:
                return None
            step = float(state["step"])
            v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)
            parts.append((v_hat.sqrt() + eps).reshape(-1).to(torch.float64))
        return torch.cat(parts)

    @torch.no_grad()
    def _epoch_sums(self):
        """Per-step and per-cell sums of the epoch, vectorised over all its steps."""
        idx, z, norms, weights = zip(*self._batches)
        sizes = torch.tensor([len(n) for n in norms], device=self.device)
        step = torch.repeat_interleave(torch.arange(len(norms), device=self.device), sizes)
        r2 = torch.cat(norms).to(torch.float64) ** 2
        if all(w is None for w in weights):
            c = r2
        else:
            w = torch.cat([torch.ones_like(n) if w is None else w for n, w in zip(norms, weights)])
            c = w.to(torch.float64) ** 2 * r2
        x = torch.cat(z).reshape(-1).to(torch.float64) / self.scale
        width = self.log_edges[1] - self.log_edges[0]
        cell = torch.floor((torch.log10(x.abs()) - self.log_edges[0]) / width).long().clamp_(0, self.n_cells - 1)
        n_steps = len(norms)
        per_step = lambda v: torch.bincount(step, weights=v, minlength=n_steps)
        n_eff = lambda v: per_step(v) ** 2 / per_step(v * v)
        out = {"mean_r2": per_step(r2) / sizes, "n_eff_w": n_eff(c), "n_eff_u": n_eff(r2),
               "sum_r2": torch.bincount(cell, weights=r2, minlength=self.n_cells),
               "sum_c": torch.bincount(cell, weights=c, minlength=self.n_cells),
               "count": torch.bincount(cell, minlength=self.n_cells),
               "tail_c": c[(x > TAIL[0]) & (x < TAIL[1])].sum()}
        if self.visits is not None:
            self.visits += torch.bincount(torch.cat(idx).cpu().long(), minlength=len(self.visits)).to(torch.int32)
        return out

    # --- per epoch ----------------------------------------------------------------------------------
    @torch.no_grad()
    def grid_residual(self, model):
        """Residual norms on the quadrature grid (numpy, float64)."""
        y, dy = value_and_derivative(model, self.grid_z)
        return residual_norm(cascade_residual(y, dy, *self.coefficients_grid.orders,
                                              forcing=self.coefficients_grid.forcing)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def end_epoch(self, trainer, trained_population=None, trained_points=None):
        """History columns of the epoch; stores the cell sums and, at the snapshot epochs, a snapshot."""
        grid_r = self.grid_residual(trainer.model)
        integrand = self.grid_weights * grid_r ** 2
        if self._batches:
            sums = self._epoch_sums()
        else:                                 # an epoch without steps (not produced by the trainer)
            empty = torch.zeros(0, dtype=torch.float64)
            sums = {"mean_r2": empty, "n_eff_w": empty, "n_eff_u": empty, "tail_c": torch.tensor(0.0),
                    "sum_r2": torch.zeros(self.n_cells, dtype=torch.float64),
                    "sum_c": torch.zeros(self.n_cells, dtype=torch.float64),
                    "count": torch.zeros(self.n_cells, dtype=torch.int64)}
        nan = float("nan")
        ratio = lambda a, b: float(a / b) if float(b) > 0 else nan
        finite_mean = lambda v: float(v[torch.isfinite(v)].mean()) if bool(torch.isfinite(v).any()) else nan
        row = {"physics_loss_unweighted": finite_mean(sums["mean_r2"]),
               "n_eff_weighted": finite_mean(sums["n_eff_w"]),
               "n_eff_unweighted": finite_mean(sums["n_eff_u"])}
        if self.has_tail:
            row["tail_share_r2"] = ratio(integrand[self.grid_tail].sum(), integrand.sum())
            row["tail_share_w2r2"] = ratio(sums["tail_c"], sums["sum_c"].sum())
        if self.gradient_every:
            start, self._epoch_gradient_start = self._epoch_gradient_start, len(self.gradients["epoch"])
            for column, key in (("grad_norm_data", "norm_data"), ("grad_norm_pde", "norm_pde"),
                                ("grad_cosine", "cosine"), ("adam_step_data", "adam_data"),
                                ("adam_step_pde", "adam_pde")):
                values = np.asarray(self.gradients[key][start:], dtype=np.float64)
                row[column] = float(np.nanmedian(values)) if np.isfinite(values).any() else nan
        if self.population_stride and self._batches:
            row.update(self._population_gradients(trainer))
        self.cells["epoch"].append(trainer.epoch)
        for key in ("sum_r2", "sum_c", "count"):
            self.cells[key].append(sums[key].cpu().clone())
        self.cells["grid_int_r2"].append(torch.as_tensor(np.bincount(self.grid_cell, weights=integrand,
                                                                     minlength=self.n_cells)))
        if trainer.epoch in self.snapshot_epochs:
            self.pending[trainer.epoch] = self.snapshot(trainer, grid_r, trained_population, trained_points)
        self._reset_epoch()
        return row

    def snapshot(self, trainer, grid_r, trained_population=None, trained_points=None):
        """Full state of the methods at the end of the current epoch, as numpy arrays.

        `trained_population`: base-point indices of the population trained on (discrete R3); `trained_points`: its
        coordinates (network variable), for every sampler with `population_points()`.
        """
        out = {"epoch": np.int64(trainer.epoch), "grid_r": grid_r}
        for key, value in trainer.model.state_dict().items():
            out[f"model.{key}"] = value.detach().cpu().numpy().copy()
        if self.visits is not None:
            out["visits"] = self.visits.numpy().copy()
        out.update(self._optimizer_state(trainer))
        weighting = trainer.point_weighting
        if hasattr(weighting, "snapshot_arrays"):
            out.update(weighting.snapshot_arrays())
        sampler = trainer.sampler
        if hasattr(sampler, "population_points"):
            out["population_x"] = self._physical(sampler.population_points())
            out["last_retained"] = np.int64(sampler.last_retained)
            if trained_points is not None:
                out["population_trained_x"] = self._physical(trained_points)
        if hasattr(sampler, "population"):          # discrete R3: indices into the base points
            out["population"] = sampler.population.detach().cpu().numpy().astype(np.int32)
            out["last_retained"] = np.int64(sampler.last_retained)
            if trained_population is not None:
                out["population_trained"] = trained_population.detach().cpu().numpy().astype(np.int32)
        return out

    @staticmethod
    def _optimizer_state(trainer):
        """Adam's moments (flattened in the order of the parameters) and the ReLoBRaLo state, as numpy."""
        out = {}
        params = [p for p in trainer.model.parameters() if p.requires_grad]
        states = [trainer.optimizer.state.get(p, {}) for p in params]
        if states and all("exp_avg" in s and "exp_avg_sq" in s for s in states):
            for key in ("exp_avg", "exp_avg_sq"):
                out[f"adam.{key}"] = torch.cat([s[key].detach().reshape(-1).to(torch.float64).cpu()
                                                for s in states]).numpy()
            out["adam.step"] = np.float64(float(states[0]["step"]))
        for key in ("weights", "previous", "initial"):
            value = getattr(trainer.balancer, key, None)
            if isinstance(value, torch.Tensor):
                out[f"balancer.{key}"] = value.detach().to(torch.float64).cpu().numpy().copy()
        return out

    def _physical(self, z):
        """Coordinates z (N, 1) of the network variable -> x = z / input_scale, float64, shape (N,)."""
        return z.detach().cpu().to(torch.float64).reshape(-1).numpy() / self.scale

    # --- files --------------------------------------------------------------------------------------
    def cell_arrays(self):
        """Contents of cells.npz: per-epoch cell sums (epochs x log_cells) and the fixed geometry."""
        stack = lambda key, dtype: (torch.stack(self.cells[key]).numpy() if self.cells[key]
                                    else np.zeros((0, self.n_cells), dtype=dtype))
        return {"epoch": np.asarray(self.cells["epoch"], dtype=np.int64),
                "sum_r2": stack("sum_r2", np.float64), "sum_c": stack("sum_c", np.float64),
                "count": stack("count", np.int64), "grid_int_r2": stack("grid_int_r2", np.float64),
                "log10_abs_x_edges": self.log_edges, "grid_x": self.grid_x, "grid_weights": self.grid_weights,
                "tail": np.asarray(TAIL)}

    def flush(self, directory):
        """Write the waiting snapshots and cells.npz into <directory>/diagnostics (temporary file + rename)."""
        target = Path(directory) / DIRECTORY
        target.mkdir(parents=True, exist_ok=True)
        for epoch in sorted(self.pending):
            _atomic_npz(target / snapshot_name(epoch), self.pending[epoch], compressed=True)
        self.pending.clear()
        _atomic_npz(target / CELLS_FILE, self.cell_arrays())
        if self.gradient_every:
            _atomic_npz(target / GRADIENTS_FILE, self.gradient_arrays())
        if self.population_stride:
            _atomic_npz(target / POPULATION_GRADIENTS_FILE, self.population_arrays(), compressed=True)

    def population_arrays(self):
        """Contents of population_gradients.npz: epochs and the three gradient vectors per epoch (float32)."""
        def stack(key):
            return torch.stack(self.population[key]).numpy() if self.population[key] else np.zeros((0, 0), np.float32)
        return {"epoch": np.asarray(self.population["epoch"], dtype=np.int64),
                "stride": np.int64(self.population_stride), **{key: stack(key) for key in POPULATION_KEYS}}

    def gradient_arrays(self):
        """Contents of gradients.npz: one row per sampled step."""
        return {key: np.asarray(values, dtype=np.int64 if key in ("epoch", "step") else np.float64)
                for key, values in self.gradients.items()}

    # --- checkpoint ---------------------------------------------------------------------------------
    def state_dict(self):
        """Cell sums so far and visit counters (snapshots not yet written are recomputed after a resume)."""
        return {"cells": {key: list(values) for key, values in self.cells.items()},
                "visits": None if self.visits is None else self.visits.clone(),
                "steps": self.steps, "gradients": {key: list(values) for key, values in self.gradients.items()},
                "population": {key: list(values) for key, values in self.population.items()}}

    def load_state_dict(self, state):
        """Restore; a checkpoint without recorder state (older run) starts the records from here."""
        self.pending = {}
        if not state:
            return
        self.cells = {key: list(values) for key, values in state["cells"].items()}
        if self.visits is not None and state.get("visits") is not None:
            self.visits = state["visits"].clone()
        self.steps = int(state.get("steps", 0))
        saved = state.get("gradients") or {}
        self.gradients = {key: list(saved.get(key, [])) for key in GRADIENT_KEYS}
        self._epoch_gradient_start = len(self.gradients["epoch"])
        saved = state.get("population") or {}
        self.population = {key: list(saved.get(key, [])) for key in ("epoch",) + POPULATION_KEYS}


def build_recorder(config, domain, input_scale, epochs):
    """`TrainingRecorder` from the `training.diagnostics` block, or None when the block is absent."""
    if config is None or config is False:
        return None
    if not isinstance(config, dict):
        raise ValueError("training.diagnostics must be a mapping (or absent)")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown keys in training.diagnostics: {sorted(unknown)} (allowed: {sorted(DEFAULTS)})")
    return TrainingRecorder(domain, input_scale, epochs, **config)
