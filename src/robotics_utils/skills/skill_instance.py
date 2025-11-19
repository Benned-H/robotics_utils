"""Define a class to represent instantiations of object-parameterized skills."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from robotics_utils.skills.skill import Skill
    from robotics_utils.skills.skills_inventory import SkillsInventory, SkillsProtocol


@dataclass(frozen=True)
class SkillInstance:
    """A skill instantiated using particular concrete arguments."""

    skill: Skill
    """Specifies the skill instance's parameter signature."""

    bindings: dict[str, object]
    """Maps each skill parameter name to its bound argument."""

    def __str__(self) -> str:
        """Return a readable string representation of the skill instance."""
        return f"{self.skill.name}({', '.join(map(str, self.arguments))})"

    @classmethod
    def from_string(
        cls,
        string: str,
        available_skills: SkillsInventory,
        universe: Mapping[str, object],
    ) -> SkillInstance:
        """Construct a SkillInstance from the given string.

        :param string: String description of a skill instance
        :param available_skills: Inventory specifying the available skills
        :param universe: Maps object names to object instances in the domain of discourse
        :return: Constructed SkillInstance instance
        """
        match = re.match(r"^(\w+)\(([^)]*)\)$", string.strip())
        if not match:
            raise ValueError(f"Could not parse SkillInstance string: '{string}'.")

        skill_name = match.group(1)
        if skill_name not in available_skills.skills:
            raise ValueError(f"Invalid skill name parsed from string: '{skill_name}'.")
        skill = available_skills.skills[skill_name]

        args_string = match.group(2).strip()
        args_names = [arg.strip() for arg in args_string.split(",")] if args_string else []

        if len(skill.parameters) != len(args_names):
            len_param = len(skill.parameters)
            raise ValueError(
                f"Skill '{skill_name}' expects {len_param} args, not {len(args_names)}.",
            )

        bindings: dict[str, object] = {}
        for bound_arg_name, param in zip(args_names, skill.parameters):  # Avoid strict=True
            if bound_arg_name not in universe:
                raise ValueError(f"Argument '{bound_arg_name}' not found in the universe.")

            bound_arg = universe[bound_arg_name]
            if not isinstance(bound_arg, param.type_):
                raise TypeError(
                    f"Cannot parse skill instance from '{string}' because skill parameter "
                    f"'{param.name}' expects type {param.type_name} but the provided "
                    f"argument '{bound_arg_name}' has type {type(bound_arg)}: {bound_arg}.",
                )

            bindings[param.name] = bound_arg

        return SkillInstance(skill, bindings)

    @property
    def arguments(self) -> tuple[object, ...]:
        """Retrieve the arguments of the skill instance in parameter order."""
        return tuple(self.bindings[param.name] for param in self.skill.parameters)

    def execute(self, executor: SkillsProtocol) -> object | None:
        """Execute this skill instance."""
        return self.skill.execute(executor, self.bindings)
