"""Optimizer and learning-rate scheduler from the `training` section of a configuration."""

import torch


class ExponentialUntil(torch.optim.lr_scheduler.ExponentialLR):
    """ExponentialLR whose lr stops decaying after `stop_epoch` epochs: the epochs from `stop_epoch` on use
    the lr of epoch `stop_epoch`, gamma^(stop_epoch - 1) times the initial one. Up to there it is the same chain of
    multiplications as ExponentialLR, so the two runs coincide bit for bit before `stop_epoch`."""

    def __init__(self, optimizer, gamma, stop_epoch, last_epoch=-1):
        if int(stop_epoch) < 1:
            raise ValueError("stop_epoch must be a positive epoch")
        self.stop_epoch = int(stop_epoch)
        super().__init__(optimizer, gamma, last_epoch)

    def get_lr(self):
        if 0 < self.last_epoch < self.stop_epoch:
            return super().get_lr()
        return [group["lr"] for group in self.optimizer.param_groups]


OPTIMIZERS = {"adam": torch.optim.Adam}
SCHEDULERS = {"exponential_until": ExponentialUntil,                                  # {gamma, stop_epoch}
              "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,    # eps: 0 lifts the 1e-8 default
              "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,                # {T_max: epochs, eta_min}
              "exponential": torch.optim.lr_scheduler.ExponentialLR}               # {gamma}


def _split(config, registry, kind):
    params = dict(config)
    name = params.pop("name")
    try:
        return registry[name], params
    except KeyError:
        raise ValueError(f"unknown {kind} {name!r}; available: {sorted(registry)}") from None


def build_optimizer(config, parameters):
    """e.g. {name: adam, lr: 1.0e-2} (Boni Table 6.1)."""
    cls, params = _split(config, OPTIMIZERS, "optimizer")
    return cls(parameters, **params)


def build_scheduler(config, optimizer):
    """e.g. {name: reduce_on_plateau, factor: 0.5, patience: 5, cooldown: 5, min_lr: 1.0e-15}."""
    cls, params = _split(config, SCHEDULERS, "scheduler")
    return cls(optimizer, **params)
