"""Define types and classes to represent facts about tuples of objects."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

ArgsT = TypeVar("ArgsT")
"""Specifies the type of the arguments expected by a constraint."""


class Constraint(ABC, Generic[ArgsT]):
    """A constraint is a Boolean function on arguments of a specific type."""

    @abstractmethod
    def __call__(self, args: ArgsT) -> bool:
        """Evaluate the constraint on the given arguments.

        :param args: Arguments on which the constraint is evaluated
        :return: True if the constraint is satisfied, else False
        """

    @property
    def name(self) -> str:
        """Retrieve the name of the constraint's class."""
        return type(self).__name__


@dataclass(frozen=True)
class Fact(Generic[ArgsT]):
    """A constraint that evaluates true on a particular tuple of arguments."""

    constraint: Constraint[ArgsT]
    satisfying_args: ArgsT

    def __post_init__(self) -> None:
        """Validate that the instantiated fact is True."""
        if not self.constraint(self.satisfying_args):
            raise ValueError(f"Unable to instantiate {self.constraint.name}.")
