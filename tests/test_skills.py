"""Unit tests for the Skill, SkillInstance, and SkillsInventory class."""

from __future__ import annotations

import pytest

from robotics_utils.abstractions import Parameter
from robotics_utils.skills import SkillInstance, SkillsInventory, SkillsProtocol, skill_method


@pytest.fixture
def example_skills_protocol() -> type[SkillsProtocol]:
    """Construct an example skills protocol as a Python class."""

    class ExampleSkills(SkillsProtocol):
        """An example skills protocol defining skills as Python methods."""

        @skill_method
        def return_true(self) -> bool:
            """Define a skill that always returns the value `True`."""
            return True

        @skill_method
        def add_one(self, value: int) -> int:
            """Define a skill that adds one to the given value.

            :param value: Integer to be incremented
            """
            return value + 1

    return ExampleSkills


def test_skills_inventory_from_protocol(example_skills_protocol: type[SkillsProtocol]) -> None:
    """Verify that a SkillsInventory is correctly constructed from an example Python protocol."""
    # Arrange/Act - An example skills protocol is provided by a pytest fixture
    inventory = SkillsInventory.from_protocol(example_skills_protocol)

    # Assert - Verify that the constructed SkillsInventory contains the expected skills
    assert inventory.name == "ExampleSkills"

    return_true = inventory.skills.get("ReturnTrue")
    assert return_true is not None
    assert return_true.parameters == ()

    add_one = inventory.skills.get("AddOne")
    assert add_one is not None
    assert add_one.parameters == (Parameter("value", int, "Integer to be incremented"),)

    assert len(inventory.all_argument_types) == 1
    assert int in inventory.all_argument_types


def test_skill_execution(example_skills_protocol: type[SkillsProtocol]) -> None:
    """Verify that skills constructed from a skills protocol can be executed."""
    # Arrange - Convert the skills protocol into an inventory of skills
    executor_instance = example_skills_protocol()  # Instantiate the protocol class
    inventory = SkillsInventory.from_protocol(example_skills_protocol)

    return_true = inventory.skills.get("ReturnTrue")
    assert return_true is not None
    add_one = inventory.skills.get("AddOne")
    assert add_one is not None

    # Act - Attempt to execute the constructed skills
    expected_true = return_true.execute(executor_instance, bindings={})
    expected_two = add_one.execute(executor_instance, bindings={"value": 1})

    # Assert
    assert expected_true
    assert expected_two == 2


@pytest.fixture
def example_skill_instance_strings() -> list[str]:
    """Generate a list of strings representing skill instances."""
    return ["ReturnTrue()", "AddOne(44)", "AddOne(-5)"]


@pytest.fixture
def example_universe() -> dict[str, object]:
    """Generate an example universe of objects."""
    return {"44": 44, "-5": -5, "100": 100}


def test_skill_instance_from_string(
    example_skill_instance_strings: list[str],
    example_skills_protocol: type[SkillsProtocol],
    example_universe: dict[str, object],
) -> None:
    """Verify that SkillInstances can be constructed from strings."""
    # Arrange - Skill instance strings, a skills protocol, and objects are provided via fixture
    inventory = SkillsInventory.from_protocol(example_skills_protocol)
    example_strings = example_skill_instance_strings

    # Act - Convert the example strings into SkillInstances
    results = [SkillInstance.from_string(s, inventory, example_universe) for s in example_strings]

    # Assert - Expect that the constructed SkillInstances have correct bindings
    return_true_instance, forty_four_instance, negative_five_instance = results
    assert return_true_instance.skill.name == "ReturnTrue"
    assert not return_true_instance.bindings

    assert forty_four_instance.skill.name == "AddOne"
    expected_44 = forty_four_instance.bindings.get("value")
    assert expected_44 == 44

    assert negative_five_instance.skill.name == "AddOne"
    expected_neg_5 = negative_five_instance.bindings.get("value")
    assert expected_neg_5 == -5
