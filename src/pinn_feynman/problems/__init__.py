"""The diagrams the pipeline can train on: one file per problem, found by name (`get_problem`).

Add a diagram by adding a file here (see `_registry.Problem` for what it declares) together with its system in
`pinn_feynman.systems` and its true values in `pinn_feynman.reference`; no other file lists the problems.
"""

from ._registry import Problem, ProblemMapping, UnknownProblem, available_problems, get_problem

__all__ = ["Problem", "ProblemMapping", "available_problems", "get_problem", "UnknownProblem"]
