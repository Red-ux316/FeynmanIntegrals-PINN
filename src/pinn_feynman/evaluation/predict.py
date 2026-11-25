"""Reload a training run and evaluate its network."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import load_config, torch_dtype
from ..data.generation import generator_info
from ..models import build_model
from ..physics import SYSTEMS, system_variables
from ..problems import available_problems, get_problem


def problem_shape(problem):
    """(number of master integrals learnt by the network, Laurent orders as a tuple, variables) of a problem, read
    from what is already registered (no table of its own, CT-07): the masters from `physics.SYSTEMS`, the variables
    from `physics.system_variables`, the orders from the reference generator (`data.generation`). Checked against the
    N_MASTERS of the reference module where it declares one. For a bottom-up problem the masters are its
    `unknowns` and the orders come from its base problem."""
    if problem not in SYSTEMS:
        raise ValueError(f"unknown problem {problem!r}; available: {available_problems()}")
    declared = get_problem(problem)
    n_masters = int(SYSTEMS[problem].N_MASTERS)
    _, info = generator_info(declared.data_problem())
    reference_masters = getattr(declared.reference_module(), "N_MASTERS", n_masters)
    if reference_masters != n_masters:
        raise ValueError(f"{problem}: {n_masters} master integrals in the system, {reference_masters} in the reference")
    if declared.unknowns is not None:
        n_masters = len(declared.unknowns)
    return n_masters, tuple(info["orders"]), system_variables(problem)


def require_one_variable(config, what):
    """ValueError unless the problem of `config` has one variable (`what` names the caller in the message)."""
    problem = config["physics"]["system"]
    variables = problem_shape(problem)[2]
    if len(variables) != 1:
        raise ValueError(f"{what}: one variable only: {problem} has the variables {variables}; for it only "
                         f"`python -m pinn_feynman.evaluation.evaluate` is available")


@dataclass
class Run:
    directory: Path
    config: dict
    part: str
    model: torch.nn.Module
    summary: dict

    @property
    def input_scale(self):
        return float(self.config["physics"]["input_scale"])

    @property
    def problem(self):
        """The problem whose data the run uses (for a bottom-up run, its base)."""
        return self.config["data"]["problem"]

    @property
    def system_problem(self):
        """The problem of the equations the network solves (`physics.system`; a bottom-up variant of `problem`)."""
        return self.config["physics"]["system"]

    @property
    def columns(self):
        """Columns of the dataset the network learns (None = all the master integrals)."""
        return get_problem(self.system_problem).columns()

    def target(self, dataset):
        """True values the network is compared with: the dataset part of the run, on its columns."""
        values = dataset.part(self.part)
        return values if self.columns is None else values[:, list(self.columns)]

    @property
    def orders(self):
        return problem_shape(self.system_problem)[1]

    @property
    def variables(self):
        return problem_shape(self.system_problem)[2]


def load_run(directory, which="best"):
    """Configuration, part and network ('best' or 'final' weights) of a run directory."""
    directory = Path(directory)
    config = load_config(directory / "config.yaml")
    n_masters, orders, _ = problem_shape(config["physics"]["system"])
    model = build_model(config["model"], n_masters, len(orders), dtype=torch_dtype(config["dtype"]), orders=orders)
    model.load_state_dict(torch.load(directory / f"{which}_model.pt", weights_only=True))
    model.eval()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return Run(directory=directory, config=config, part=config["part"], model=model, summary=summary)


@torch.no_grad()
def predict(model, x, input_scale, batch=20_000):
    """Network output at the points x (numpy, shape (N,) for one variable, (N, d) for d variables, in the
    physical variables) -> numpy (N, n_masters, n_orders). The network input is input_scale * x."""
    dtype = next(model.parameters()).dtype
    x = np.asarray(x)
    d = 1 if x.ndim == 1 else x.shape[-1]
    z = torch.as_tensor(x * input_scale, dtype=dtype).reshape(-1, d)
    return torch.cat([model(chunk) for chunk in torch.split(z, batch)]).numpy()
