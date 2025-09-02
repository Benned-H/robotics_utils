"""Define a class to represent Python-typed symbolic parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.predicates.dataclass_type import DataclassType


# TODO: Delete other parameters.py file once this works as expected
@dataclass(frozen=True)
class Parameter:
    """A Python-typed symbolic parameter (e.g., of a predicate or operator)."""

    name: str
    """Name of the lifted parameter."""

    type_: type
    """Python type expected by the parameter."""

    semantics: str | None = None
    """Optional natural language description of the parameter's meaning."""

    def __str__(self) -> str:
        """Create a readable string representation of the parameter."""
        semantics = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (type {self.type_}){semantics}"

    @classmethod
    def tuple_from_dataclass(cls, dataclass_t: DataclassType) -> tuple[Parameter, ...]:
        """Construct a tuple of Parameter instances from a Python dataclass type."""
        docstrings = dataclass_t.get_docstrings()
        return tuple(
            Parameter(f_name, f_type, docstrings[f_name])
            for f_name, f_type in dataclass_t.field_types.items()
        )
