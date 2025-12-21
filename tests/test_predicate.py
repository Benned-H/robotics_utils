"""Unit tests for the Predicate class."""

from __future__ import annotations

import pytest

from robotics_utils.abstractions.symbols import DiscreteParameter, ObjectSymbol, Predicate


@pytest.fixture
def available() -> Predicate:
    """Define a predicate named `available` expecting one parameter."""
    h = DiscreteParameter(name="?h", type_="hoist")
    return Predicate("available", (h,))


@pytest.fixture
def lifting() -> Predicate:
    """Define a predicate named `lifting` expecting two parameters."""
    h = DiscreteParameter(name="?h", type_="hoist")
    c = DiscreteParameter(name="?c", type_="crate")
    return Predicate("lifting", (h, c))


def test_predicate_to_pddl(available: Predicate, lifting: Predicate) -> None:
    """Verify that the Predicate class can reproduce PDDL predicates from a real PDDL domain."""
    # Arrange - Fixtures define predicates analogous to those in the IPC-5 "Storage" domain

    # Act - Convert the predicates into their PDDL string representations
    available_pddl = available.to_pddl()
    lifting_pddl = lifting.to_pddl()

    # Assert - Expect that the PDDL exactly matches the domain text
    assert available_pddl == "(available ?h - hoist)"
    assert lifting_pddl == "(lifting ?h - hoist ?c - crate)"


def test_predicate_grounding(lifting: Predicate) -> None:
    """Verify that a Predicate can be fully grounded using parameter bindings."""
    # Arrange - Define bindings for the two parameters of the predicate
    blue_crate = ObjectSymbol(name="blue_crate", type_="crate")
    red_hoist = ObjectSymbol(name="red_hoist", type_="hoist")

    bindings = {"?c": blue_crate, "?h": red_hoist}

    # Act - Ground the predicate using the bindings
    lifting_ground_atom = lifting.fully_bind(bindings)

    # Assert - Expect that the arguments of the predicate instance are correct
    expected_args = (red_hoist, blue_crate)
    assert expected_args == lifting_ground_atom.arguments
