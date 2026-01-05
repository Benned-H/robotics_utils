"""Define classes to represent Python-typed parameters of skills, predicates, etc."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generic, Hashable, Iterator, TypeVar, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

DataclassT = TypeVar("DataclassT", bound=Hashable)
"""Represents a type of hashable dataclass."""


class DataclassType(Generic[DataclassT]):
    """Utility class wrapper for a type of hashable dataclass."""

    def __init__(self, dataclass_t: type[DataclassT]) -> None:
        """Initialize the class using the given dataclass type."""
        if not isinstance(dataclass_t, type) or not is_dataclass(dataclass_t):
            raise TypeError(f"{dataclass_t} is not a dataclass type.")
        self._dataclass_t: type[DataclassInstance] = dataclass_t

    @property
    def field_names(self) -> Iterator[str]:
        """Provide an iterator over the field names of the dataclass type."""
        for field in fields(self._dataclass_t):
            yield field.name

    @cached_property
    def field_types(self) -> dict[str, type]:
        """Retrieve a mapping from dataclass field names to the corresponding types."""
        type_hints = get_type_hints(self._dataclass_t)
        return {f.name: type_hints[f.name] for f in fields(self._dataclass_t)}

    def get_docstrings(self) -> dict[str, str | None]:
        """Retrieve a mapping from field names to metadata docstrings, if any (else None)."""
        return {f.name: f.metadata.get("doc") for f in fields(self._dataclass_t)}

    def new(self, **kwargs: object) -> DataclassInstance:
        """Construct an instance of the stored dataclass type."""
        return self._dataclass_t(**kwargs)


@dataclass(frozen=True)
class Parameter:
    """A Python-typed symbolic parameter (e.g., of a skill or predicate)."""

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
