"""Unit tests for the unification algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from robotics_utils.fol.unification import Unifiable, UnifierBindings, unify
from robotics_utils.predicates import Parameter, Predicate


class Person(str): ...


class AlwaysHoldsPredicate(Predicate):
    """An example predicate that holds in all states for all arguments."""

    def holds_in(self, state: Any, args: Any) -> bool:
        """Evaluate whether the predicate holds in the given state for the given arguments."""
        return True


@dataclass(frozen=True)
class KnowsArgs:
    """Arguments to the `Knows` predicate representing if one person knows another."""

    y: Person
    x: Person


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

    @dataclass(frozen=True)
    class KnowsArgs:
        """Arguments to the `Knows` predicate representing if one person knows another."""

        y: Person
        x: Person

    y_knows_x = AlwaysHoldsPredicate.from_dataclass("Knows", KnowsArgs)

    john = Person("John")
    jane = Person("Jane")
    bill = Person("Bill")
    elizabeth = Person("Elizabeth")

    john_knows_x = y_knows_x.as_atom({"y": john})
    john_knows_jane = y_knows_x.fully_ground({"y": john, "x": jane})

    y_knows_bill = y_knows_x.as_atom({"x": bill})

    @dataclass(frozen=True)
    class ReverseKnowsArgs:
        """Reversed arguments of the `Knows` predicate."""

        x: Person
        y: Person

    x_knows_y = AlwaysHoldsPredicate.from_dataclass("Knows", ReverseKnowsArgs)

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
