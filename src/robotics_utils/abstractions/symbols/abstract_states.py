"""Define classes to represent symbolic abstract states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.ground_atom import GroundAtom
    from robotics_utils.abstractions.symbols.objects import Objects
    from robotics_utils.abstractions.symbols.predicate import Predicate


@dataclass(frozen=True)
class AbstractState:
    """An abstract state is the set of ground atoms that hold in a low-level state."""

    facts: frozenset[GroundAtom]

    def __contains__(self, ground_atom: GroundAtom) -> bool:
        """Evaluate whether a given ground atom is in the abstract state's facts."""
        return ground_atom in self.facts

    def __str__(self) -> str:
        """Create a human-readable string representation of the abstract state."""
        sorted_facts = "\n\t".join(sorted(str(fact) for fact in self.facts))
        return f"AbstractState(\n\t{sorted_facts}\n)"


class AbstractStateSpace:
    """An abstract state space specifies all possible ground atoms in a planning problem."""

    def __init__(self, predicates: Iterable[Predicate], objects: Objects) -> None:
        """Initialize the abstract state space using all valid groundings of the given predicates.

        :param predicates: Predicates defining possible abstract relations between objects
        :param objects: Symbols representing concrete objects in the environment
        """
        self.possible_facts: set[GroundAtom] = set()
        for p in predicates:
            ground_atoms = p.compute_all_groundings(objects)
            self.possible_facts.update(ground_atoms)
