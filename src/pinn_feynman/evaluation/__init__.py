"""Evaluation of trained networks: errors (Boni eqs. 6.18-6.19), figures, comparison with the thesis."""

from .metrics import absolute_error, relative_error, summarize
from .predict import load_run, predict

__all__ = ["absolute_error", "load_run", "predict", "relative_error", "summarize"]
