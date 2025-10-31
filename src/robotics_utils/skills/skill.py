"""Define a class to represent object-parameterized skills."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Mapping, get_type_hints

from robotics_utils.abstractions.predicates import Parameter
from robotics_utils.io.string_utils import is_pascal_case, pascal_to_snake, snake_to_pascal
from robotics_utils.meta import parse_docstring_params
from robotics_utils.skills.skill_instance import SkillInstance

if TYPE_CHECKING:
    from robotics_utils.skills.skills_inventory import SkillsProtocol


def skill_method(m: Callable) -> Callable:
    """Mark a method as implementing a skill."""
    m._is_skill = True
    return m


@dataclass(frozen=True)
class Skill:
    """A skill parameterized by Python-typed arguments."""

    name: str
    """A skill's name should be PascalCase (e.g., "OpenDoor")."""

    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        """Validate expected properties of any Skill instance."""
        if not is_pascal_case(self.name):
            raise ValueError(f"Skill name '{self.name}' must be PascalCase.")

    def __str__(self) -> str:
        """Return a readable string representation of the skill."""
        params = ", ".join(f"{p.name}: {p.type_name}" for p in self.parameters)
        return f"{self.name}({params})"

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

            # Get parameter semantics from the method docstring
            semantics = param_docs.get(param_name)
            if semantics is None:
                raise ValueError(f"Skill {skill_name} didn't define semantics for '{param_name}'.")

            parameters.append(Parameter(param_name, param_type, semantics))

        return Skill(skill_name, tuple(parameters))

    @property
    def method_name(self) -> str:
        """Retrieve the method name corresponding to this skill."""
        return pascal_to_snake(self.name)  # PascalCase skill name -> snake_case method name

    def execute(self, executor: SkillsProtocol, bindings: Mapping[str, object]) -> object | None:
        """Execute this skill under the given parameter bindings.

        :param executor: Protocol defining an interface for skill execution
        :param bindings: Map from parameter names to bound arguments
        """
        if not hasattr(executor, self.method_name):
            raise NotImplementedError(f"Skills protocol has no method: {self.method_name}.")

        skill_method = getattr(executor, self.method_name)
        args = [bindings[param.name] for param in self.parameters]
        return skill_method(executor, *args)

    def create_all_instances(self, objects: list[object]) -> list[SkillInstance]:
        """Compute all valid instantiations of the skill using the given Python objects.

        :param objects: Collection of Python object instances
        :return: List of all valid instances of the skill
        """
        objects_per_param_type = [
            [obj for obj in objects if isinstance(obj, param.type_)] for param in self.parameters
        ]

        # Find all valid tuples of concrete args using a Cartesian product
        all_valid_args = product(*objects_per_param_type)

        return [
            SkillInstance(
                self,
                bindings={p.name: obj for p, obj in zip(self.parameters, args, strict=True)},
            )
            for args in all_valid_args
        ]
