"""Unit tests for the unification algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from robotics_utils.classical_planning import DiscreteParameter, Predicate
from robotics_utils.fol.unification import Unifiable, UnifierBindings, unify


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
    person_x = DiscreteParameter("x", object_type="Person")
    person_y = DiscreteParameter("y", object_type="Person")

    y_knows_x = Predicate("Knows", parameters=(person_y, person_x))

    john_knows_x = y_knows_x.as_atom({"y": "John"})
    john_knows_jane = y_knows_x.fully_ground({"y": "John", "x": "Jane"})

    y_knows_bill = y_knows_x.as_atom({"x": "Bill"})

    x_knows_y = Predicate("Knows", parameters=(person_x, person_y))

    x_knows_elizabeth = x_knows_y.as_atom({"y": "Elizabeth"})

    return [
        UnificationTestCase(john_knows_x, john_knows_jane, {"x": "Jane"}),
        UnificationTestCase(john_knows_x, y_knows_bill, {"x": "Bill", "y": "John"}),
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
