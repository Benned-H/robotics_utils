"""Define a dataclass to represent PDDL problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from robotics_utils.classical_planning.abstract_states import AbstractState
from robotics_utils.classical_planning.fol import Sentence
from robotics_utils.classical_planning.parameters import ObjectT
from robotics_utils.classical_planning.pddl_domain import PDDLDomain


@dataclass(frozen=True)
class PDDLProblem(Generic[ObjectT]):
    """A PDDL problem describes the concrete details of a particular planning problem."""

    name: str
    """Name of the PDDL problem."""

    domain: PDDLDomain
    """PDDL domain corresponding to this problem."""

    objects: dict[ObjectT, set[str]]
    """A map from objects in the environment to their sets of types."""

    initial_state: AbstractState
    """The initial state (i.e., set of true grounded predicates) in the planning problem."""

    goal_condition: Sentence
    """A logical expression that must be satisfied at the end of any valid solution plan."""
