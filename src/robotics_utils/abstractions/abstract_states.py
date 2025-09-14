"""Define classes to represent and compute symbolic abstract states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

StateT = TypeVar("StateT")
"""Represents a low-level environment state."""

if TYPE_CHECKING:
    from robotics_utils.abstractions.predicates import Predicate, PredicateInstance


@dataclass(frozen=True)
class AbstractState(Generic[StateT]):
    """An abstract state is the set of predicate instances (i.e., facts) true in a state."""

    facts: set[PredicateInstance[Any, StateT]]

    def __contains__(self, predicate_instance: PredicateInstance) -> bool:
        """Evaluate whether a given predicate instance is in the abstract state's facts."""
        return predicate_instance in self.facts

    def __str__(self) -> str:
        """Create a readable string representation of the abstract state."""
        sorted_facts = "\n\t".join(sorted(str(fact) for fact in self.facts))
        return f"AbstractState(\n\t{sorted_facts}\n)"

    def __sub__(self, other: AbstractState[StateT]) -> AbstractState[StateT]:
        """Compute the set difference between the facts in this and another abstract state."""
        if not isinstance(other, AbstractState):
            raise TypeError(f"Unsupported operand type: {type(other)}.")
        return AbstractState(self.facts - other.facts)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the abstract state."""
        all_facts = "\n\t".join(sorted(fact.to_pddl() for fact in self.facts))
        return f"(and\n\t{all_facts}\n)"


class AbstractStateSpace(Generic[StateT]):
    """An abstract state space specifies all possible predicate instances in the abstract state."""

    def __init__(
        self,
        predicates: Iterable[Predicate[Any, StateT]],
        objects: Iterable[object],
    ) -> None:
        """Initialize the abstract state space using all valid groundings of the given predicates.

        :param predicates: Predicates defining possible abstract relations between objects
        :param objects: Collection of Python objects in the environment
        """
        self.possible_facts: set[PredicateInstance[Any, StateT]] = set()
        for predicate in predicates:
            all_instances = predicate.compute_all_groundings(objects)
            self.possible_facts.update(all_instances)

    def abstract(self, state: StateT) -> AbstractState:
        """Compute the abstract state for the given low-level state.

        :param state: Low-level state of the environment
        :return: Computed abstract state (i.e., all facts that are true in the low-level state)
        """
        return AbstractState({fact for fact in self.possible_facts if fact.holds_in(state)})
