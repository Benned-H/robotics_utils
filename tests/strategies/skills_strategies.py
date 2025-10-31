"""Define strategies for generating skill-related objects for property-based testing."""

from __future__ import annotations

import hypothesis.strategies as st

from robotics_utils.abstractions.predicates import Parameter
from robotics_utils.skills import Skill, SkillsInventory

from .common_strategies import pascal_case_strings


@st.composite
def generate_parameters(draw: st.DrawFn) -> Parameter:
    """Generate random Python-typed parameters."""
    name = draw(st.text(min_size=1))
    param_type = draw(st.one_of([st.just(str), st.just(int), st.just(float), st.just(bool)]))
    semantics = draw(st.one_of(st.none(), st.text()))
    return Parameter(name, param_type, semantics)


@st.composite
def generate_parameters_tuple(draw: st.DrawFn) -> tuple[Parameter, ...]:
    """Generate a tuple of Parameters with unique names."""
    parameter_names = draw(st.lists(st.text(min_size=1), unique=True))

    parameters = []
    for p_name in parameter_names:
        p_type = draw(st.one_of([st.just(str), st.just(int), st.just(float), st.just(bool)]))
        p_semantics = draw(st.one_of(st.none(), st.text()))
        parameters.append(Parameter(p_name, p_type, p_semantics))

    return tuple(parameters)


@st.composite
def generate_skills(draw: st.DrawFn) -> Skill:
    """Generate random object-centric skills."""
    name = draw(pascal_case_strings())  # Skill names should be Pascal case
    parameters = draw(generate_parameters_tuple())
    return Skill(name, parameters)


@st.composite
def generate_skills_inventories(draw: st.DrawFn) -> SkillsInventory:
    """Generate random inventories of skills."""
    inventory_name = draw(st.text())

    skill_names = draw(st.lists(pascal_case_strings(), unique=True))
    skills = [Skill(name, parameters=draw(generate_parameters_tuple())) for name in skill_names]

    return SkillsInventory(inventory_name, skills)
