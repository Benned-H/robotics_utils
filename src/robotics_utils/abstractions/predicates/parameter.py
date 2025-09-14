"""Define a class to represent Python-typed symbolic parameters."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_utils.abstractions.predicates.dataclass_type import DataclassT, DataclassType


@dataclass(frozen=True)
class Parameter:
    """A Python-typed symbolic parameter (e.g., of a predicate or operator)."""

    name: str
    """Name of the lifted parameter."""

    type_: type
    """Python type expected by the parameter."""

    semantics: str | None = None
    """Optional natural language description of the parameter's meaning."""

    def __post_init__(self) -> None:
        """Verify required properties for any constructed Parameter instance."""
        if not isinstance(self.type_, type):
            raise TypeError(f"{self.name}: type_ must be a class or type, not {self.type_!r}.")

        try:
            isinstance(None, self.type_)
        except TypeError as err:
            raise TypeError(
                f"{self.name}: type_ {self.type_!r} doesn't support isinstance() checks.",
            ) from err

        try:
            hash(self.type_)
        except TypeError as err:
            raise TypeError(f"{self.name}: type_ {self.type_!r} doesn't support hashing.") from err

    def __str__(self) -> str:
        """Create a readable string representation of the parameter."""
        semantics = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (type {self.type_name}){semantics}"

    @classmethod
    def tuple_from_dataclass(cls, dataclass_t: type[DataclassT]) -> tuple[Parameter, ...]:
        """Construct a tuple of Parameter instances from a Python dataclass type."""
        dataclass_type = DataclassType(dataclass_t)
        docstrings = dataclass_type.get_docstrings()

        return tuple(
            Parameter(f_name, f_type, docstrings[f_name])
            for f_name, f_type in dataclass_type.field_types.items()
        )

    @property
    def type_name(self) -> str:
        """Retrieve a human-readable string name for the parameter's type."""
        return self.type_.__name__
