"""Unit tests for classes representing lifted and grounded predicates."""

import pytest

from robotics_utils.classical_planning import DiscreteParameter, Predicate


@pytest.fixture
def hoist_h() -> DiscreteParameter:
    """Define a discrete parameter named `?h` of type `hoist`."""
    return DiscreteParameter(name="?h", object_type="hoist")


@pytest.fixture
def crate_c() -> DiscreteParameter:
    """Define a discrete parameter named `?c` of type `crate`."""
    return DiscreteParameter(name="?c", object_type="crate")


@pytest.fixture
def store_area_s() -> DiscreteParameter:
    """Define a discrete parameter named `?s` of type `storearea`."""
    return DiscreteParameter(name="?s", object_type="storearea")


@pytest.fixture
def available(hoist_h: DiscreteParameter) -> Predicate:
    """Define a predicate named `available` expecting one parameter."""
    return Predicate(name="available", parameters=(hoist_h,))


@pytest.fixture
def lifting(hoist_h: DiscreteParameter, crate_c: DiscreteParameter) -> Predicate:
    """Define a predicate named `lifting` expecting two parameters."""
    return Predicate(name="lifting", parameters=(hoist_h, crate_c))


@pytest.fixture
def on(crate_c: DiscreteParameter, store_area_s: DiscreteParameter) -> Predicate:
    """Define a predicate named `on` expecting two parameters."""
    return Predicate(name="on", parameters=(crate_c, store_area_s))


def test_predicate_to_pddl(available: Predicate, lifting: Predicate, on: Predicate) -> None:
    """Verify that the Predicate class can reproduce PDDL predicates from a real PDDL domain."""
    # Arrange - Fixtures define three predicates analogous to those in the IPC-5 "Storage" domain

    # Act - Convert the predicates into their PDDL string representation
    available_pddl = available.to_pddl()
    lifting_pddl = lifting.to_pddl()
    on_pddl = on.to_pddl()

    # Assert - Expect that the predicates' PDDL exactly matches the domain text
    assert available_pddl == "(available ?h - hoist)"
    assert lifting_pddl == "(lifting ?h - hoist ?c - crate)"
    assert on_pddl == "(on ?c - crate ?s - storearea)"


def test_predicate_ground_with(lifting: Predicate) -> None:
    """Verify that a Predicate instance can be grounded using parameter bindings."""
    # Arrange - Define bindings for the two parameters of the `lifting` predicate
    bindings = {"?c": "blue_crate", "?h": "red_hoist"}

    # Act - Ground the predicate using the bindings
    lifting_instance = lifting.ground_with(bindings=bindings)

    # Assert - Expect that the arguments of the predicate instance are correct
    expected_args = ("red_hoist", "blue_crate")  # Recall: `lifting` put the hoist before the crate
    assert expected_args == lifting_instance.arguments
