"""Define a class representing an inventory of skills available to an agent."""

from __future__ import annotations

from types import FunctionType
from typing import Any, Iterator, Tuple

from robotics_utils.io.string_utils import snake_to_pascal
from robotics_utils.meta import get_default_values
from robotics_utils.skills.skill import Skill

SkillsProtocol = object
"""Represents an arbitrary protocol defining skills for a particular domain."""

SkillParamKey = Tuple[str, str]
"""A key identifying a skill parameter using the tuple (skill name, parameter name)."""


def get_skill_methods(protocol: SkillsProtocol) -> list[FunctionType]:
    """Find all skill methods specified by the given protocol."""
    all_methods = [getattr(protocol, method_name) for method_name in dir(protocol)]
    return [method for method in all_methods if hasattr(method, "_is_skill")]


def find_default_param_values(protocol: SkillsProtocol) -> dict[SkillParamKey, Any]:
    """Find default parameter values (where specified) for skills in the given protocol.

    :param protocol: Python protocol specifying skill signatures
    :return: Map from (skill name, param name) keys to that parameter's default value
    """
    default_param_values = {}
    for method in get_skill_methods(protocol):
        skill_name = snake_to_pascal(method.__name__)  # Skill names are PascalCase
        for param_name, default_value in get_default_values(method).items():
            default_param_values[(skill_name, param_name)] = default_value

    return default_param_values


class SkillsInventory:
    """An inventory of skills available to an agent."""

    def __init__(self, name: str, skills: list[Skill]) -> None:
        """Initialize the skills inventory using the given list of skills."""
        self.name = name
        self.skills: dict[str, Skill] = {s.name: s for s in skills}
        """Map from skill names to Skill instances."""

        self.all_argument_types: set[type] = {p.type_ for skill in skills for p in skill.parameters}
        """Set of argument types used by the skills inventory."""

    def __str__(self) -> str:
        """Create a readable string representation of the skills inventory."""
        skill_names = sorted(self.skills.keys())
        skill_strings = [str(self.skills[name]) for name in skill_names]
        return self.name + "\n\t".join(skill_strings)

    def __key(self) -> tuple[str, tuple[Skill, ...]]:
        """Define a key to uniquely identify the SkillsInventory."""
        skill_names = sorted(self.skills.keys())
        sorted_skills = tuple(self.skills[name] for name in skill_names)
        return (self.name, sorted_skills)

    def __hash__(self) -> int:
        """Generate a hash value for the SkillsInventory."""
        return hash(self.__key())

    def __eq__(self, other: object) -> bool:
        """Evaluate whether another SkillsInventory is equal to this one."""
        if isinstance(other, SkillsInventory):
            return self.__key() == other.__key()
        return NotImplemented

    def __iter__(self) -> Iterator[Skill]:
        """Provide an iterator over the skills in the inventory."""
        yield from self.skills.values()

    @classmethod
    def from_protocol(cls, protocol: SkillsProtocol) -> SkillsInventory:
        """Extract a skills inventory from the methods of a Python protocol.

        :param protocol: Python protocol specifying skill signatures
        :return: Constructed SkillsInventory instance
        """
        skill_methods = get_skill_methods(protocol)
        skills = [Skill.from_method(method) for method in skill_methods]

        return SkillsInventory(name=type(protocol).__name__, skills=skills)

    def get_skill(self, skill_name: str) -> Skill:
        """Retrieve the named skill from the inventory."""
        if skill_name not in self.skills:
            raise KeyError(f"Cannot retrieve skill with unknown name: '{skill_name}'.")
        return self.skills[skill_name]
