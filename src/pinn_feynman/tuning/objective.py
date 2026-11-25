"""Objective of a tuning trial: short training, best validation loss, pruning."""

import optuna
import torch

from ..training.setup import build_experiment
from .space import suggest, trial_config


class PruningCallback:
    """Report the validation loss to Optuna after each epoch; stop the trial if pruned."""

    def __init__(self, trial):
        self.trial = trial

    def __call__(self, trainer):
        self.trial.report(trainer.history.rows[-1]["val_loss"], step=trainer.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {trainer.epoch}")


def make_objective(base_config, budget, space, part, datasets):
    """Objective(trial) -> best validation loss of a short training with the trial's values.

    Non-finite losses (a diverged trial) are reported as pruned.
    """
    def objective(trial):
        values = suggest(trial, space)
        config = trial_config(base_config, budget, values)
        trial.set_user_attr("hidden", config["model"]["hidden"])
        torch.set_num_threads(config["training"].get("threads", 1))
        trainer = build_experiment(config, part, datasets=datasets).trainer
        trainer.fit(config["training"]["epochs"], on_epoch_end=PruningCallback(trial),
                    desc=f"trial {trial.number}", leave=False)
        if not torch.isfinite(torch.tensor(trainer.best_val)):
            raise optuna.TrialPruned("non-finite validation loss")
        trial.set_user_attr("best_epoch", trainer.best_epoch)
        return trainer.best_val

    return objective
