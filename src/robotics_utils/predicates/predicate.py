"""Define a class to represent lifted symbolic predicates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from robotics_utils.predicates.parameters import ArgsT

StateT = TypeVar("StateT")  # TODO: Move to appropriate file


@dataclass(frozen=True)
class Predicate(ABC, Generic[StateT, ArgsT]):
    """A symbolic predicate with typed parameters."""

    # TODO: Member variables

    @abstractmethod
    def holds_in(self, state: StateT, args: ArgsT) -> bool:
        """Evaluate whether the predicate holds in a state for the given arguments.

        :param state: Low-level environment state in which the predicate is evaluated
        :param args: Arguments bound to the predicate's parameters
        :return: True if the grounded predicate holds, else False
        """

    # TODO: Other methods


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT, ArgsT]):
    """A predicate grounded using particular concrete arguments."""

    predicate: Predicate
    arguments: ArgsT

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.holds_in(state, self.arguments)
