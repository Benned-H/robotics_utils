"""Unit tests for the Skill and SkillInstance classes."""

from hypothesis import given

from robotics_utils.skills.skills import Skill

from .skills_strategies import skills


@given(skills())
def test_skill_to_from_yaml_data(skill: Skill) -> None:
    """Verify that Skills are accurately reconstructed after converting to and from YAML data."""
    # Arrange/Act - Given a skill, convert to YAML data and then convert back
    result_yaml = skill.to_yaml_data()
    result_skill = Skill.from_yaml_data(skill_name=skill.name, yaml_data=result_yaml)

    # Assert - Expect that the resulting Skill is identical to the original
    assert skill == result_skill
