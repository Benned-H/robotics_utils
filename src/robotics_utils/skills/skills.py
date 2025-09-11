"""Define classes to represent object-parameterized skills and their instantiations."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from robotics_utils.classical_planning.parameters import Bindings, DiscreteParameter
from robotics_utils.io.process_python import parse_docstring_params
from robotics_utils.io.string_utils import is_pascal_case, pascal_to_snake, snake_to_pascal

if TYPE_CHECKING:
    from robotics_utils.classical_planning.objects import Objects
    from robotics_utils.skills.skills_inventory import SkillsInventory, SkillsProtocol


def skill_method(m: Callable) -> Callable:
    """Mark a method as implementing a skill."""
    m._is_skill = True
    return m


@dataclass(frozen=True)
class Skill:
    """A skill parameterized by object-typed arguments."""

    name: str
    """A skill's name should be PascalCase (e.g., "OpenDoor")."""

    parameters: tuple[DiscreteParameter, ...]

    def __post_init__(self) -> None:
        """Validate expected properties of any Skill instance."""
        if not is_pascal_case(self.name):
            raise ValueError(f"Skill name '{self.name}' must be PascalCase.")

    def __str__(self) -> str:
        """Return a readable string representation of the skill."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    @classmethod
    def from_yaml_data(cls, skill_name: str, skill_data: dict[str, Any]) -> Skill:
        """Load a Skill instance from data imported from YAML."""
        if skill_name not in skill_data:
            raise KeyError(f"Skill name '{skill_name}' missing from YAML data: {skill_data}.")

        if "parameters" not in skill_data[skill_name]:
            raise KeyError(f"Key 'parameters' missing from skill YAML data: {skill_data}.")
        params_data = skill_data[skill_name]["parameters"]

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
        skill_name = snake_to_pascal(method.__name__)
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

            object_type = param_type.__name__

            # Get parameter semantics from the method docstring
            semantics = param_docs.get(param_name)
            if semantics is None:
                raise ValueError(f"Skill {skill_name} didn't define semantics for '{param_name}'.")

            parameters.append(DiscreteParameter(param_name, object_type, semantics))

        return Skill(skill_name, tuple(parameters))

    def execute(self, executor: SkillsProtocol, bindings: Bindings) -> object | None:
        """Execute this skill under the given object bindings.

        :param executor: Protocol defining an interface for skill execution
        :param bindings: Map from parameter names to bound object names
        """
        method_name = pascal_to_snake(self.name)  # PascalCase skill name -> snake_case method name

        if not hasattr(executor, method_name):
            raise NotImplementedError(f"Skills protocol has no method: {method_name}")

        skill_method = getattr(executor, method_name)
        args = [bindings[param.name] for param in self.parameters]
        return skill_method(executor, *args)
