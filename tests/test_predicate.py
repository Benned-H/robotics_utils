"""Unit tests for the Predicate class.

Reference: storage-domain.pddl from https://lpg.unibs.it/ipc-5/generators/index.html
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from robotics_utils.abstractions.symbols import DiscreteParameter, ObjectSymbol, Predicate


@pytest.fixture
def available() -> Predicate:
    """Define a predicate named `available` expecting one parameter."""
    h = DiscreteParameter(name="h", type_="hoist")
    return Predicate("available", (h,))


@pytest.fixture
def lifting() -> Predicate:
    """Define a predicate named `lifting` expecting two parameters."""
    h = DiscreteParameter(name="h", type_="hoist")
    c = DiscreteParameter(name="c", type_="crate")
    return Predicate("lifting", (h, c))


@pytest.fixture
def on() -> Predicate:
    """Define a predicate named `on` expecting two parameters."""
    c = DiscreteParameter(name="c", type_="crate")
    s = DiscreteParameter(name="s", type_="storearea")
    return Predicate("on", (c, s))


def test_predicate_to_pddl(available: Predicate, lifting: Predicate, on: Predicate) -> None:
    """Verify that the Predicate class can reproduce PDDL predicates from a real PDDL domain."""
    # Arrange - Fixtures define three predicates analogous to those in the IPC-5 "Storage" domain

    # Act - Convert the predicates into their PDDL string representations
    available_pddl = available.to_pddl()
    lifting_pddl = lifting.to_pddl()
    on_pddl = on.to_pddl()

    # Assert - Expect that the PDDL exactly matches the domain text
    assert available_pddl == "(available ?h - hoist)"
    assert lifting_pddl == "(lifting ?h - hoist ?c - crate)"
    assert on_pddl == "(on ?c - crate ?s - storearea)"


def test_predicate_instantiation(lifting: Predicate) -> None:
    """Verify that a Predicate can be fully grounded using parameter bindings."""
    # Arrange - Define bindings for the two parameters of the `lifting` predicate
    red_hoist = ObjectSymbol(name="red_hoist", type_="hoist")
    blue_crate = ObjectSymbol(name="blue_crate", type_="crate")

    bindings = {"c": blue_crate, "h": red_hoist}

    # Act - Ground the predicate using the bindings
    ground_atom = lifting.fully_bind(bindings)

    # Assert - Expect that the arguments of the predicate instance are correct
    assert ground_atom.arguments == (red_hoist, blue_crate)
