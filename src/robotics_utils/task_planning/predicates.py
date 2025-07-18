"""Define classes to represent symbolic predicates, lifted and grounded."""

from __future__ import annotations

import re
from dataclasses import dataclass

from robotics_utils.task_planning.parameters import Bindings, DiscreteParameter


@dataclass(frozen=True)
class Predicate:
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    semantics: str | None = None  # Optional NL description of the predicate's meaning

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    @classmethod
    def from_pddl(cls, pddl: str) -> Predicate:
        """Construct a predicate from a string of PDDL.

        :param pddl: PDDL string representing a predicate
        :return: Constructed instance of the Predicate class
        """
        match = re.match(r"^\((\S+)(.*)\)$", pddl.strip())
        if not match:
            raise ValueError(f"Could not parse Predicate from PDDL string: '{pddl}'")

        name = match.group(1).strip()
        params_string = match.group(2)

        # Process the parameters string to identify the parameters and their types
        parameters: list[DiscreteParameter] = []  # Parameters that have been finalized
        awaiting_type: list[str] = []  # Parameter names waiting for their type to be specified
        next_token_is_type = False  # Indicates that the next token should be a parameter type

        for token in params_string.split():
            if next_token_is_type:
                type_name = token
                parameters.extend(DiscreteParameter(param, type_name) for param in awaiting_type)

                next_token_is_type = False
                awaiting_type = []

            elif token == "-":
                next_token_is_type = True
            else:
                awaiting_type.append(token)

        if awaiting_type:
            error = f"Predicate '{name}' didn't define a type for parameters: {awaiting_type}."
            raise ValueError(error)

        return Predicate(name, tuple(parameters))

    @classmethod
    def from_operator_pddl(cls, pddl: str, predicates: dict[str, Predicate]) -> Predicate:
        """Construct a predicate from a string of PDDL within an operator definition.

        Example: In an operator, we might see a predicate such as (Stacked ?on_top ?on_bottom).

        :param pddl: String referencing the predicate from within a PDDL operator
        :param predicates: Collection of predicates available in the PDDL domain
        :return: Constructed instance of the Predicate class
        """
        match = re.match(r"^\((\w+)(.*)\)$", pddl.strip())
        if not match:
            raise ValueError(f"Could not parse predicate string: '{pddl}'")

        name = match.group(1).strip()
        if name not in predicates:
            raise ValueError(f"Predicate '{name}' is unknown in set: {predicates}")
        existing_params = predicates[name].parameters

        params_string = match.group(2)
        new_param_names = params_string.split() if params_string else []
        if len(new_param_names) != len(existing_params):
            raise ValueError(
                f"Parsed {len(new_param_names)} parameters for predicate {name} "
                f"but the existing predicate expects {len(existing_params)}.",
            )
        new_params = [
            DiscreteParameter(new_name, existing_param.object_type, existing_param.semantics)
            for existing_param, new_name in zip(existing_params, new_param_names, strict=True)
        ]

        return Predicate(name, tuple(new_params), predicates[name].semantics)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        types_to_params: dict[str, list[str]] = {}  # Map type names to all such predicate params
        for param in self.parameters:
            if param.object_type not in types_to_params:
                types_to_params[param.object_type] = []
            types_to_params[param.object_type].append(param.name)

        type_groups = []
        for type_name, relevant_params in types_to_params.items():
            pddl_params = " ".join(relevant_params)
            type_groups.append(f"{pddl_params} - {type_name}")

        params_string = " " + " ".join(type_groups) if type_groups else ""
        return f"({self.name}{params_string})"


@dataclass(frozen=True)
class PredicateInstance:
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate
    bindings: Bindings

    def __str__(self) -> str:
        """Return a readable string representation of the predicate instance."""
        args_string = ", ".join(self.bindings[p.name] for p in self.predicate.parameters)
        return f"{self.predicate.name}({args_string})"

    @property
    def name(self) -> str:
        """Retrieve the name of the predicate instance."""
        return self.predicate.name

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate instance."""
        args_string = " ".join(self.bindings[p.name] for p in self.predicate.parameters)
        return f"({self.predicate.name} {args_string})"
