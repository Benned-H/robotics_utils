"""Unit tests for the Skill, SkillInstance, and SkillsInventory class."""

from __future__ import annotations

import pytest
from hypothesis import given

from robotics_utils.classical_planning.objects import Objects
from robotics_utils.skills.skills import Skill, SkillInstance, skill_method
from robotics_utils.skills.skills_inventory import SkillsInventory, SkillsProtocol

from .skills_strategies import generate_skills, generate_skills_inventories


@given(generate_skills())
def test_skill_to_from_yaml_data(skill: Skill) -> None:
    """Verify that Skills are accurately reconstructed after converting to and from YAML data."""
    # Arrange/Act - Given a skill, convert to YAML data and then convert back
    result_yaml = skill.to_yaml_data()
    result_skill = Skill.from_yaml_data(skill.name, result_yaml)

    # Assert - Expect that the resulting Skill is identical to the original
    assert skill == result_skill


@pytest.fixture
def example_skills_protocol() -> SkillsProtocol:
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

            :param value: Integer to be increased by 1
            """
            return value + 1

    return ExampleSkills


@given(generate_skills_inventories())
def test_skills_inventory_to_from_yaml_data(inventory: SkillsInventory) -> None:
    """Verify that a SkillsInventory is unchanged after converting to and from YAML data."""
    # Arrange/Act - Given an inventory of skills, convert to YAML data and then convert back
    inventory_yaml = inventory.to_yaml_data()
    result_inventory = SkillsInventory.from_yaml_data(inventory.name, inventory_yaml)

    # Assert - Expect that the resulting SkillsInventory is identical to the original
    assert inventory == result_inventory


def test_skills_inventory_from_protocol(example_skills_protocol: SkillsProtocol) -> None:
    """Verify that a SkillsInventory is correctly constructed from an example Python protocol."""
    # Arrange/Act - An example skills protocol is provided by a pytest fixture
    inventory = SkillsInventory.from_protocol(example_skills_protocol)

    # Assert - Verify that the constructed SkillsInventory contains the expected skills
    assert inventory.name == "ExampleSkills"

    return_true = inventory.skills.get("ReturnTrue")
    assert return_true is not None

    add_one = inventory.skills.get("AddOne")
    assert add_one is not None

    assert len(inventory.object_types) == 1
    assert "int" in inventory.object_types


def test_skill_execution(example_skills_protocol: SkillsProtocol) -> None:
    """Verify that skills constructed from a skills protocol can be executed."""
    # Arrange - Convert the skills protocol into an inventory of skills
    inventory = SkillsInventory.from_protocol(example_skills_protocol)

    return_true = inventory.skills.get("ReturnTrue")
    assert return_true is not None
    add_one = inventory.skills.get("AddOne")
    assert add_one is not None

    # Act - Attempt to execute the constructed skills
    expected_true = return_true.execute(example_skills_protocol, bindings={})
    expected_two = add_one.execute(example_skills_protocol, bindings={"value": 1})

    # Assert
    assert expected_true
    assert expected_two == 2


@pytest.fixture
def example_skill_instance_strings() -> list[str]:
    """Generate a list of strings representing skill instances."""
    return ["ReturnTrue()", "AddOne(44)", "AddOne(-5)"]


@pytest.fixture
def example_objects() -> Objects:
    """Generate an example collection of objects."""
    object_to_types = {"44": {"int", "truthy"}, "-5": {"int", "negative"}}
    return Objects(object_to_types)


def test_skill_instance_from_string(
    example_skill_instance_strings: list[str],
    example_skills_protocol: SkillsProtocol,
    example_objects: Objects,
) -> None:
    """Verify that SkillInstances can be constructed from strings."""
    # Arrange - Skill instance strings, a skills protocol, and objects are provided via fixture
    inventory = SkillsInventory.from_protocol(example_skills_protocol)
    example_strings = example_skill_instance_strings

    # Act - Convert the example strings into SkillInstances
    results = [SkillInstance.from_string(s, inventory, example_objects) for s in example_strings]

    # Assert - Expect that the constructed SkillInstances have correct bindings
    return_true_instance, forty_four_instance, negative_five_instance = results
    assert return_true_instance.skill.name == "ReturnTrue"
    assert not return_true_instance.bindings

    assert forty_four_instance.skill.name == "AddOne"
    expected_44 = forty_four_instance.bindings.get("value")
    assert expected_44 == "44"

    assert negative_five_instance.skill.name == "AddOne"
    expected_neg_5 = negative_five_instance.bindings.get("value")
    assert expected_neg_5 == "-5"
