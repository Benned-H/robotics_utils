"""Define a class to represent operators (i.e., lifted abstract action templates)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from robotics_utils.abstractions.operators.operator_instance import OperatorInstance

if TYPE_CHECKING:
    from robotics_utils.abstractions.operators.effects import Effects
    from robotics_utils.abstractions.operators.preconditions import Preconditions
    from robotics_utils.abstractions.predicates import Parameter


@dataclass(frozen=True)
class Operator:
    """A lifted abstract action defining an abstract transition model."""

    name: str
    parameters: tuple[Parameter, ...]
    preconditions: Preconditions  # Positive and negative preconditions for applying the operator
    effects: Effects  # Effects added and removed from the abstract state by the operator

    def ground_with(self, bindings: Mapping[str, Any]) -> OperatorInstance:
        """Ground the operator using the given parameter bindings."""
        return OperatorInstance(self, bindings)
