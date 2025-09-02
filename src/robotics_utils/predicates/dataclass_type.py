"""Define a class representing a dataclass type."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from inspect import getdoc
from typing import Any, Generic, Iterator, TypeVar

DataclassT = TypeVar("DataclassT")


class DataclassType(Generic[DataclassT]):
    """A dataclass type (i.e., not an instance)."""

    def __init__(self, dataclass_t: type[DataclassT]) -> None:
        """Initialize the class using the given dataclass type."""
        if not isinstance(dataclass_t, type) or not is_dataclass(dataclass_t):
            raise TypeError(f"{dataclass_t} is not a dataclass type.")
        self._dataclass_t = dataclass_t

    @property
    def field_names(self) -> Iterator[str]:
        """Provide an iterator over the field names of the dataclass type."""
        for field in fields(self._dataclass_t):
            yield field.name

    @property
    def field_types(self) -> dict[str, Any]:
        """Retrieve a mapping from dataclass field names to the corresponding types."""
        return {f.name: f.type for f in fields(self._dataclass_t)}

    def get_docstrings(self) -> dict[str, str | None]:
        """Retrieve a mapping from field names to corresponding docstrings, if any (else None)."""
        return {f.name: (getdoc(f) if getdoc(f) else None) for f in fields(self._dataclass_t)}
