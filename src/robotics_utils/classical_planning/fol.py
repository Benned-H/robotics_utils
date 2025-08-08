"""Define classes to represent sentences in first-order logic (FOL)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

StateT_contra = TypeVar("StateT_contra", contravariant=True)
"""Represents a state of the world used to evaluate the truth value of a ground FOL expression."""


class Sentence(Protocol, Generic[StateT_contra]):
    """A logical sentence whose truth value can be determined in a given state."""

    def __call__(self, state: StateT_contra) -> bool:
        """Evaluate the truth value of the sentence in the given state."""
        ...


@dataclass(frozen=True)
class Negation(Sentence[StateT_contra]):
    """A negation has the opposite truth value of the sentence it negates."""

    operand: Sentence[StateT_contra]

    def __call__(self, state: StateT_contra) -> bool:
        """Evaluate the negation in a low-level state.

        :param state: Low-level state of the environment
        :return: True if the operand sentence is false in the state, else False
        """
        return not self.operand(state)


@dataclass(frozen=True)
class Conjunction(Sentence[StateT_contra]):
    """A conjunction (i.e., AND) is only true if all sentences in the conjunction are true."""

    operands: set[Sentence[StateT_contra]]

    def __call__(self, state: StateT_contra) -> bool:
        """Evaluate the conjunction in a low-level state.

        :param state: Low-level state of the environment
        :return: True if all sentences in the conjunction are true, else False
        """
        return all(sentence(state) for sentence in self.operands)


@dataclass(frozen=True)
class Disjunction(Sentence[StateT_contra]):
    """A disjunction (i.e., OR) is true if any sentence in the disjunction is true."""

    operands: set[Sentence[StateT_contra]]

    def __call__(self, state: StateT_contra) -> bool:
        """Evaluate the disjunction in a low-level state.

        :param state: Low-level state of the environment
        :return: True if any sentence in the disjunction is true, else False
        """
        return any(sentence(state) for sentence in self.operands)


@dataclass(frozen=True)
class Implication(Sentence[StateT_contra]):
    """An implication is true unless its premise is true but its conclusion is not."""

    premise: Sentence[StateT_contra]
    conclusion: Sentence[StateT_contra]

    def __call__(self, state: StateT_contra) -> bool:
        """Evaluate the implication in a low-level state.

        :param state: Low-level state of the environment
        :return: True, unless the premise is true in the state yet the conclusion is false
        """
        return (not self.premise(state)) or self.conclusion(state)


@dataclass(frozen=True)
class Biconditional(Sentence[StateT_contra]):
    """A biconditional is true if and only if its operands are logically equivalent."""

    operand_a: Sentence[StateT_contra]
    operand_b: Sentence[StateT_contra]

    def __call__(self, state: StateT_contra) -> bool:
        """Evaluate the biconditional in a low-level state.

        :param state: Low-level state of the environment
        :return: True if the operands have the same truth value in the state, else False
        """
        return self.operand_a(state) == self.operand_b(state)


Not = Negation
And = Conjunction
Or = Disjunction
Implies = Implication
IFF = Biconditional
