"""Define a class representing a dataclass type."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Hashable, Iterator, TypeVar, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

DataclassT = TypeVar("DataclassT", bound=Hashable)
"""A type variable representing any hashable dataclass type."""


class DataclassType(Generic[DataclassT]):
    """A dataclass type (i.e., not an instance)."""

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

    def new(self, **kwargs: object) -> DataclassT:
        """Construct an instance of the stored dataclass type."""
        return self._dataclass_t(**kwargs)
