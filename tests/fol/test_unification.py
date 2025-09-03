"""Unit tests for the unification algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from robotics_utils.fol.unification import Unifiable, UnifierBindings, unify
from robotics_utils.predicates import Predicate


class Person(str): ...


StringState = str  # Stands in for an actual state representation


@dataclass(frozen=True)
class KnowsArgs:
    """Arguments to the `Knows` predicate representing if one person knows another."""

    y: Person
    x: Person


def knows_person(_args: KnowsArgs, _state: StringState) -> bool:
    """Evaluate whether one person knows another in a given state."""
    return True


@dataclass(frozen=True)
class ReversedKnowsArgs:
    """Reversed arguments of the `Knows` predicate."""

    x: Person
    y: Person


def reversed_knows_person(_args: ReversedKnowsArgs, _state: StringState) -> bool:
    """Evaluate whether one person knows another in a given state."""
    return True


@dataclass(frozen=True)
class UnificationTestCase:
    """A pair of FOL expressions and the bindings that unify them (or None if impossible)."""

    x: Unifiable
    y: Unifiable
    expected_bindings: UnifierBindings | None


@pytest.fixture
def unification_test_cases() -> list[UnificationTestCase]:
    """Construct and return a collection of unification test cases.

    Reference: Examples from Section 9.2.1 ("Unification") of AIMA (4th Ed.) by Russell and Norvig.
    """
    y_knows_x = Predicate("Knows", KnowsArgs, relation=knows_person)

    john = Person("John")
    jane = Person("Jane")
    bill = Person("Bill")
    elizabeth = Person("Elizabeth")

    john_knows_x = y_knows_x.as_atom({"y": john})
    john_knows_jane = y_knows_x.fully_ground({"y": john, "x": jane})

    y_knows_bill = y_knows_x.as_atom({"x": bill})

    x_knows_y = Predicate("Knows", ReversedKnowsArgs, relation=reversed_knows_person)

    x_knows_elizabeth = x_knows_y.as_atom({"y": elizabeth})

    return [
        UnificationTestCase(john_knows_x, john_knows_jane, {"x": jane}),
        UnificationTestCase(john_knows_x, y_knows_bill, {"x": bill, "y": john}),
        UnificationTestCase(john_knows_x, x_knows_elizabeth, expected_bindings=None),
    ]


def test_unification_examples(unification_test_cases: list[UnificationTestCase]) -> None:
    """Verify that unification produces correct substitutions for example FOL expressions."""
    # Arrange - Test cases are provided by a fixture

    # Act/Assert - Unify each test case and check for the correct result
    for test_case in unification_test_cases:
        result_bindings = unify(test_case.x, test_case.y, bindings={})
        if test_case.expected_bindings is None:
            assert result_bindings is None
        else:
            assert result_bindings == test_case.expected_bindings
