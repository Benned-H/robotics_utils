"""Define classes to represent symbolic abstract states."""

from dataclasses import dataclass

from robotics_utils.classical_planning.predicates import PredicateInstance


@dataclass(frozen=True)
class AbstractState:
    """An abstract state is the set of true predicate instances (i.e., facts) in a state."""

    facts: set[PredicateInstance]

    def __contains__(self, predicate_instance: PredicateInstance) -> bool:
        """Evaluate whether a given predicate instance is in the abstract state's facts."""
        return predicate_instance in self.facts

    def __str__(self) -> str:
        """Create a readable string representation of the abstract state."""
        sorted_facts = "\n\t".join(sorted(str(fact) for fact in self.facts))
        return f"AbstractState(\n\t{sorted_facts}\n)"
