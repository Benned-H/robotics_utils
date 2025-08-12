"""Define a class representing an inventory of skills available to an agent."""

from __future__ import annotations

from typing import Any

from robotics_utils.skills.skills import Skill, SkillsProtocol


class SkillInventory:
    """An inventory of skills available to an agent."""

    def __init__(self, name: str, skills: list[Skill]) -> None:
        """Initialize the skill inventory using the given list of skills."""
        self.name = name
        self.skills: dict[str, Skill] = {s.name: s for s in skills}
        """Map from skill names to Skill instances."""

        self.object_types: set[str] = {p.object_type for skill in skills for p in skill.parameters}
        """Set of object types used by the skill inventory."""

    @classmethod
    def from_protocol(cls, protocol: SkillsProtocol) -> SkillInventory:
        """Extract a skill inventory from the methods of a Python protocol.

        :param protocol: Python protocol specifying skill signatures
        :return: Constructed SkillInventory instance
        """
        methods = [getattr(protocol, method_name) for method_name in dir(protocol)]
        skills = [Skill.from_method(method) for method in methods if hasattr(method, "_is_skill")]

        return SkillInventory(name=protocol.__name__, skills=skills)

    @classmethod
    def from_yaml_data(cls, inventory_name: str, yaml_data: dict[str, Any]) -> SkillInventory:
        """Import a SkillInventory instance from a YAML file.

        :param inventory_name: Name given to the constructed skill inventory
        :param yaml_data: Filepath to a YAML file containing skills and type data
        :return: Constructed SkillInventory instance
        """
        skills = [Skill.from_yaml_data(name, data) for name, data in yaml_data["skills"].items()]
        return SkillInventory(inventory_name, skills)

    def to_yaml_data(self) -> dict[str, Any]:
        """Convert the skill inventory into a dictionary ready to be exported as YAML data."""
        skills_data = {}
        for skill in self.skills.values():
            skills_data.update(skill.to_yaml_data())

        types_data = sorted(self.object_types)

        return {"skills": skills_data, "types": types_data}


#     def __post_init__(self) -> None:
#         """Verify that the Domain is valid with respect to the following properties.

#         Valid domains must:
#             - Define at least one skill and one object type
#             - Use all defined object types in at least one skill
#             - Define all object types used by any skill parameter
#         """
#         # Verify that the domain contains at least one skill and one object type
#         if not self.skills:
#             raise ValueError(f"Domain '{self.name}' doesn't define any skills.")

#         if not self.object_types:
#             raise ValueError(f"Domain '{self.name}' doesn't specify any object types.")

#         # Compute which object types are used by some skill in the domain
#         used_types: set[str] = set()
#         for skill in self.skills.values():
#             for param in skill.parameters:
#                 used_types.add(param.object_type)

#         # Verify that all defined object types are used by at least one skill
#         unused_types = sorted(self.object_types - used_types)
#         if unused_types:
#             raise ValueError(
#                 f"Domain '{self.name}' defines unused object types: {unused_types}.\n"
#                 "These types are declared in the domain but not used by any skill.",
#             )

#         # Verify that all skills only use types defined in the domain
#         undefined_types = sorted(used_types - self.object_types)
#         if undefined_types:
#             raise ValueError(
#                 f"Skills in domain '{self.name}' use undefined object types: {undefined_types}.\n"
#                 "Add these types to the `object_types` set or fix typos in skill signatures.",
#             )
