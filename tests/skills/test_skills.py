"""Unit tests for the Skill and SkillInstance classes."""

import hypothesis.strategies as st
from hypothesis import given

from robotics_utils.classical_planning.parameters import DiscreteParameter
from robotics_utils.skills.skill_inventory import SkillInventory
from robotics_utils.skills.skills import Skill

from ..common_strategies import camel_case_strings


@st.composite
def parameters(draw: st.DrawFn) -> DiscreteParameter:
    """Generate random object-typed discrete parameters."""
    name = draw(st.text())
    object_type = draw(st.text())
    semantics = draw(st.one_of(st.none(), st.text()))
    return DiscreteParameter(name, object_type, semantics)


@st.composite
def skills(draw: st.DrawFn) -> Skill:
    """Generate random object-centric skills."""
    name = draw(camel_case_strings())  # Skill names should be camel-case
    params_list = draw(st.lists(parameters()))

    return Skill(name, tuple(params_list))


@st.composite
def skill_inventories(draw: st.DrawFn) -> SkillInventory:
    """Generate random inventories of skills."""
    inventory_name = draw(st.text())
    skills_list = draw(st.lists(skills()))

    return SkillInventory(inventory_name, skills_list)


@given(skills())
def test_skill_to_from_yaml_data(skill: Skill) -> None:
    """Verify that Skills are accurately reconstructed after converting to and from YAML data."""
    # Arrange/Act - Given a skill, convert to YAML data and then convert back
    result_yaml = skill.to_yaml_data()
    result_skill = Skill.from_yaml_data(skill_name=skill.name, yaml_data=result_yaml)

    # Assert - Expect that the resulting Skill is identical to the original
    assert skill == result_skill


@given(skill_inventories())
def test_skill_inventory_to_from_yaml_data(inventory: SkillInventory) -> None:
    """Verify that a SkillInventory is unchanged after converting to and from YAML data."""
    # Arrange/Act - Given an inventory of skills, convert to YAML data and then convert back
    inventory_yaml = inventory.to_yaml_data()
    result_inventory = SkillInventory.from_yaml_data(inventory.name, inventory_yaml)

    # Assert - Expect that the resulting SkillInventory is identical to the original
    assert inventory == result_inventory
