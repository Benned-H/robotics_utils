"""Define a class to represent symbolic abstract states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.classical_planning.predicates import PredicateInstance


@dataclass(frozen=True)
class AbstractState:
    """An abstract state is the set of predicate instances (i.e., facts) true in a state."""

    facts: set[PredicateInstance]

    def __contains__(self, predicate_instance: PredicateInstance) -> bool:
        """Evaluate whether a given predicate instance is in the abstract state's facts."""
        return predicate_instance in self.facts

    def __str__(self) -> str:
        """Create a readable string representation of the abstract state."""
        sorted_facts = "\n\t".join(sorted(str(fact) for fact in self.facts))
        return f"AbstractState(\n\t{sorted_facts}\n)"

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the abstract state."""
        all_facts = "\n\t".join(sorted(fact.to_pddl() for fact in self.facts))
        return f"(and\n\t{all_facts}\n)"
