"""Define classes to represent object-parameterized skills and their instantiations."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from robotics_utils.classical_planning.parameters import Bindings, DiscreteParameter
from robotics_utils.io.process_python import parse_docstring_params
from robotics_utils.io.string_utils import camel_to_snake, is_camel_case, snake_to_camel

if TYPE_CHECKING:
    from robotics_utils.classical_planning.objects import Objects
    from robotics_utils.skills.skill_inventory import SkillInventory


SkillsProtocol = Any
"""Represents arbitrary skill protocols for different domains."""


def skill(func: Callable) -> Callable:
    """Mark a function as implementing a skill."""
    func._is_skill = True
    return func


@dataclass(frozen=True)
class Skill:
    """A skill parameterized by object-typed arguments."""

    name: str
    """A skill's name should be CamelCase (e.g., "OpenDoor")."""

    parameters: tuple[DiscreteParameter, ...]

    def __post_init__(self) -> None:
        """Validate expected properties of any Skill instance."""
        if not is_camel_case(self.name):
            raise ValueError(f"Skill name '{self.name}' must be CamelCase.")

    def __str__(self) -> str:
        """Return a readable string representation of the skill."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    @classmethod
    def from_yaml_data(cls, skill_name: str, yaml_data: dict[str, Any]) -> Skill:
        """Load a Skill instance from data imported from YAML."""
        if skill_name not in yaml_data:
            raise KeyError(f"Skill name '{skill_name}' missing from YAML data: {yaml_data}.")

        params_data = yaml_data[skill_name].get("parameters")
        if params_data is None:
            raise KeyError(f"Key 'parameters' missing from YAML data: {yaml_data}.")
        return Skill(skill_name, DiscreteParameter.tuple_from_yaml_data(params_data))

    def to_yaml_data(self) -> dict[str, Any]:
        """Convert the Skill object into a dictionary of YAML data."""
        params_data = {}
        for p in self.parameters:
            params_data.update(p.to_yaml_data())

        return {self.name: {"parameters": params_data}}

    @classmethod
    def from_method(cls, method: Callable[[Any], Any]) -> Skill:
        """Construct a Skill instance from a Python protocol method.

        :param method: Method defining the parameter signature of the skill
        :return: Constructed Skill instance
        """
        skill_name = snake_to_camel(method.__name__)
        method_params = inspect.signature(method).parameters
        type_hints = get_type_hints(method)

        # Parse the docstring for parameter descriptions
        docstring = inspect.getdoc(method) or ""
        param_docs = parse_docstring_params(docstring)

        parameters = []
        for param_name in method_params:
            if param_name == "self":
                continue  # Skip 'self' parameter

            # Get the parameter object type from the type hints
            param_type = type_hints.get(param_name)
            if param_type is None:
                raise ValueError(f"Skill {skill_name} didn't define a type for '{param_name}'.")

            object_type = param_type.__name__.capitalize()

            # Get parameter semantics from the method docstring
            semantics = param_docs.get(param_name)
            if semantics is None:
                raise ValueError(f"Skill {skill_name} didn't define semantics for '{param_name}'.")

            parameters.append(DiscreteParameter(param_name, object_type, semantics))

        return Skill(skill_name, tuple(parameters))

    def execute(self, executor: SkillsProtocol, bindings: Bindings) -> None:
        """Execute this skill under the given object bindings.

        :param executor: Protocol defining an interface for skill execution
        :param bindings: Map from parameter names to bound object names
        """
        method_name = camel_to_snake(self.name)  # CamelCase skill name -> snake_case method name

        if not hasattr(executor, method_name):
            raise NotImplementedError(f"Skills protocol has no method: {method_name}")

        skill_method = getattr(executor, method_name)
        args = [bindings[param.name] for param in self.parameters]
        skill_method(*args)


@dataclass(frozen=True)
class SkillInstance:
    """A skill instantiated using particular concrete objects."""

    skill: Skill
    """Specifies the skill instance's parameter signature."""

    bindings: Bindings
    """Maps each skill parameter name to the name of its bound object."""

    def __str__(self) -> str:
        """Return a readable string representation of the skill instance."""
        args_string = ", ".join(self.bindings[p.name] for p in self.skill.parameters)
        return f"{self.skill.name}({args_string})"

    @classmethod
    def from_string(
        cls,
        string: str,
        available_skills: SkillInventory,
        objects: Objects,
    ) -> SkillInstance:
        """Construct a SkillInstance from the given string.

        :param string: String description of a skill instance
        :param available_skills: Inventory specifying the available skills
        :param objects: Collection of all objects in the environment
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
        args = [arg.strip() for arg in args_string.split(",")] if args_string else []

        if len(skill.parameters) != len(args):
            len_param = len(skill.parameters)
            raise ValueError(f"Skill '{skill_name}' expects {len_param} args, not {len(args)}.")

        bindings: Bindings = {}
        for bound_object, param in zip(args, skill.parameters, strict=True):
            if bound_object not in objects:
                raise ValueError(f"Object '{bound_object}' not found in the environment.")

            obj_types = objects.get_types_of(bound_object)

            if param.object_type not in obj_types:
                raise ValueError(
                    f"Cannot parse skill instance from '{string}' because skill parameter "
                    f"'{param.name}' expects type {param.object_type} but the provided "
                    f"argument object '{bound_object}' only has type(s) {obj_types}.",
                )
            bindings[param.name] = bound_object

        return SkillInstance(skill, bindings)

    def execute(self, executor: SkillsProtocol) -> None:
        """Execute this skill instance."""
        self.skill.execute(executor, self.bindings)
