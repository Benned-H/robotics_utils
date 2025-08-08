"""Define classes to represent facts, both lifted and grounded, about object tuples."""

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

ParamsT = TypeVar("ParamsT")
"""Specifies the type of a tuple of data expected by a constraint."""

Constraint = Callable[[ParamsT], bool]
"""A constraint is a Boolean function expecting a tuple of inputs of particular types."""


@dataclass(frozen=True)
class Fact(Generic[ParamsT]):
    """A constraint that evaluates true on a particular tuple of objects."""

    constraint: Constraint[ParamsT]
    satisfying_args: ParamsT
    """A tuple of concrete arguments that satisfy the constraint."""


@dataclass(frozen=True)
class Literal(Generic[ParamsT]):
    """A literal is a fact or a negated fact."""

    fact: Fact[ParamsT]
    negated: bool


State = set[Fact[Any]]
"""A state is a set of facts. By the closed world assumption, facts not explicitly
specified in a state are false. Therefore, we do not include negated facts in the state."""
