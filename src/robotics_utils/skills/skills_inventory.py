"""Define a class representing an inventory of skills available to an agent."""

from __future__ import annotations

from robotics_utils.skills.skill import Skill

SkillsProtocol = object
"""Represents an arbitrary protocol defining skills for a particular domain."""


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

    @classmethod
    def from_protocol(cls, protocol: SkillsProtocol) -> SkillsInventory:
        """Extract a skills inventory from the methods of a Python protocol.

        :param protocol: Python protocol specifying skill signatures
        :return: Constructed SkillsInventory instance
        """
        methods = [getattr(protocol, method_name) for method_name in dir(protocol)]
        skills = [Skill.from_method(method) for method in methods if hasattr(method, "_is_skill")]

        return SkillsInventory(name=protocol.__name__, skills=skills)
