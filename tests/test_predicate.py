"""Unit tests for the Predicate class."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from robotics_utils.abstractions.objects import BaseObjectType
from robotics_utils.abstractions.predicates import Predicate


class Hoist(BaseObjectType): ...


class Crate(BaseObjectType): ...


class StorageArea(BaseObjectType): ...


StringState = str  # Stands in for an actual state representation


@dataclass(frozen=True)
class AvailableArgs:
    """Arguments of the `available` predicate."""

    h: Hoist


def is_available(_args: AvailableArgs, _state: StringState) -> bool:
    """Evaluate whether a hoist in available in a given state."""
    return True


@pytest.fixture
def available() -> Predicate:
    """Define a predicate named `available` expecting one parameter."""
    return Predicate("available", AvailableArgs, relation=is_available)


@dataclass(frozen=True)
class LiftingArgs:
    """Arguments of the `lifting` predicate."""

    h: Hoist
    c: Crate


def is_lifting(_args: LiftingArgs, _state: StringState) -> bool:
    """Evaluate whether a hoist is lifting a crate in a given state."""
    return True


@pytest.fixture
def lifting() -> Predicate:
    """Define a predicate named `lifting` expecting two parameters."""
    return Predicate("lifting", LiftingArgs, relation=is_lifting)


@dataclass(frozen=True)
class OnArgs:
    """Arguments of the `on` predicate."""

    c: Crate
    s: StorageArea


def is_on(_args: OnArgs, _state: StringState) -> bool:
    """Evaluate whether a crate is on a storage area in a given state."""
    return True


@pytest.fixture
def on() -> Predicate:
    """Define a predicate named `on` expecting two parameters."""
    return Predicate("on", OnArgs, relation=is_on)


def test_predicate_to_pddl(available: Predicate, lifting: Predicate, on: Predicate) -> None:
    """Verify that the Predicate class can reproduce PDDL predicates from a real PDDL domain."""
    # Arrange - Fixtures define three predicates analogous to those in the IPC-5 "Storage" domain

    # Act - Convert the predicates into their PDDL string representations
    available_pddl = available.to_pddl()
    lifting_pddl = lifting.to_pddl()
    on_pddl = on.to_pddl()

    # Assert - Expect that the PDDL exactly matches the domain text (modulo capitalization)
    assert available_pddl == "(available ?h - Hoist)"
    assert lifting_pddl == "(lifting ?h - Hoist ?c - Crate)"
    assert on_pddl == "(on ?c - Crate ?s - StorageArea)"


def test_predicate_instantiation(lifting: Predicate) -> None:
    """Verify that a Predicate can be fully grounded using parameter bindings."""
    # Arrange - Define bindings for the two parameters of the predicate
    blue_crate = Crate("blue_crate")
    red_hoist = Hoist("red_hoist")

    bindings = {"c": blue_crate, "h": red_hoist}

    # Act - Ground the predicate using the bindings
    lifting_instance = lifting.fully_bind(bindings)

    # Assert - Expect that the arguments of the predicate instance are correct
    expected_args = LiftingArgs(h=red_hoist, c=blue_crate)
    assert expected_args == lifting_instance.arguments
