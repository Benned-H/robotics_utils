"""Unit tests for skill definitions for the Dorfl robot."""

from __future__ import annotations

import pytest

from robotics_utils.skills import SkillsInventory, SkillsProtocol

from .examples.dorfl_skills import Bread, DorflSkillsProtocol, Jar, Knife


@pytest.fixture
def dorfl_skills_protocol() -> SkillsProtocol:
    """Retrieve the skills protocol for the Dorfl robot."""
    return DorflSkillsProtocol


@pytest.fixture
def dorfl_domain_objects() -> list[object]:
    """Construct and return the objects comprising the Dorfl "Spread PB" domain."""
    knife = Knife("knife", visual_description="Metal knife with red plastic handle")
    pb_jar = Jar("pb_jar", visual_description="Jar of peanut butter with a white label")
    bread = Bread("bread", visual_description="Slice of white bread")
    return [knife, pb_jar, bread]


def test_dorfl_skill_instances(
    dorfl_skills_protocol: SkillsProtocol,
    dorfl_domain_objects: list[object],
) -> None:
    """Verify that the Dorfl skills protocol can be converted into the expected skill instances."""
    # Arrange/Act - Convert the protocol into a skills inventory before instantiating the skills
    dorfl_skills_inventory = SkillsInventory.from_protocol(dorfl_skills_protocol)

    # Assert 1 - Expect to find the four skills in the domain
    grasp_skill = dorfl_skills_inventory.get_skill("Grasp")
    assert grasp_skill is not None

    spread_skill = dorfl_skills_inventory.get_skill("Spread")
    assert spread_skill is not None

    scoop_skill = dorfl_skills_inventory.get_skill("Scoop")
    assert scoop_skill is not None

    open_skill = dorfl_skills_inventory.get_skill("Open")
    assert open_skill is not None

    # Act - Instantiate the skills using the given environment objects
    grasp_instances = grasp_skill.create_all_instances(dorfl_domain_objects)
    spread_instances = spread_skill.create_all_instances(dorfl_domain_objects)
    scoop_instances = scoop_skill.create_all_instances(dorfl_domain_objects)
    open_instances = open_skill.create_all_instances(dorfl_domain_objects)

    # Assert 2 - Expect that all skills have one instance except Grasp, which has two
    assert len(grasp_instances) == 2
    assert len(spread_instances) == 1
    assert len(scoop_instances) == 1
    assert len(open_instances) == 1
