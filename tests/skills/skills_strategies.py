"""Define strategies for generating skill-related objects for property-based testing."""

import hypothesis.strategies as st

from robotics_utils.classical_planning.parameters import DiscreteParameter
from robotics_utils.skills.skills import Skill
from robotics_utils.skills.skills_inventory import SkillsInventory

from ..common_strategies import camel_case_strings


@st.composite
def generate_parameters(draw: st.DrawFn) -> DiscreteParameter:
    """Generate random object-typed discrete parameters."""
    name = draw(st.text())
    object_type = draw(st.text())
    semantics = draw(st.one_of(st.none(), st.text()))
    return DiscreteParameter(name, object_type, semantics)


@st.composite
def generate_skills(draw: st.DrawFn) -> Skill:
    """Generate random object-centric skills."""
    name = draw(camel_case_strings())  # Skill names should be camel-case
    params_list = draw(st.lists(generate_parameters()))

    return Skill(name, tuple(params_list))


@st.composite
def generate_skills_inventories(draw: st.DrawFn) -> SkillsInventory:
    """Generate random inventories of skills."""
    inventory_name = draw(st.text())
    skills_list = draw(st.lists(generate_skills()))

    return SkillsInventory(inventory_name, skills_list)
