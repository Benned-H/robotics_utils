"""Unit tests for the Predicate class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from robotics_utils.predicates import Predicate


class Hoist(str):
    """Represents a hoist in a PDDL problem."""

    __slots__ = ()


class Crate(str):
    """Represents a crate in a PDDL problem."""

    __slots__ = ()


class StorageArea(str):
    """Represents a storage area in a PDDL problem."""

    __slots__ = ()


@pytest.fixture
def example_predicate() -> type[Predicate]:
    """Create an example Predicate class."""

    class AlwaysHolds(Predicate):
        """An example predicate implementing the abstract `holds_in` method."""

        def holds_in(self, state: Any, args: Any) -> bool:
            """Evaluate whether the predicate holds in the given state for the given arguments."""
            return True

    return AlwaysHolds


@dataclass(frozen=True)
class AvailableArgs:
    """Arguments of the `available` predicate."""

    h: Hoist


@pytest.fixture
def available(example_predicate: type[Predicate]) -> Predicate:
    """Define a predicate named `available` expecting one parameter."""
    return example_predicate.from_dataclass(name="available", dataclass_t=AvailableArgs)


@dataclass(frozen=True)
class LiftingArgs:
    """Arguments of the `lifting` predicate."""

    h: Hoist
    c: Crate


@pytest.fixture
def lifting(example_predicate: type[Predicate]) -> Predicate:
    """Define a predicate named `lifting` expecting two parameters."""
    return example_predicate.from_dataclass(name="lifting", dataclass_t=LiftingArgs)


@dataclass(frozen=True)
class OnArgs:
    """Arguments of the `on` predicate."""

    c: Crate
    s: StorageArea


@pytest.fixture
def on(example_predicate: type[Predicate]) -> Predicate:
    """Define a predicate named `on` expecting two parameters."""
    return example_predicate.from_dataclass(name="on", dataclass_t=OnArgs)


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


def test_predicate_fully_grounded(lifting: Predicate) -> None:
    """Verify that a Predicate can be fully grounded using parameter bindings."""
    # Arrange - Define bindings for the two parameters of the `lifting` predicate
    blue_crate = Crate("blue_crate")
    red_hoist = Hoist("red_hoist")

    bindings = {"c": blue_crate, "h": red_hoist}

    # Act - Ground the predicate using the bindings
    lifting_instance = lifting.fully_ground(bindings)

    # Assert - Expect that the arguments of the predicate instance are correct
    expected_args = LiftingArgs(h=red_hoist, c=blue_crate)
    assert expected_args == lifting_instance.arguments
