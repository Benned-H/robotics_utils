"""Define a class representing a first-order logic symbol denoting a typed object."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectSymbol:
    """A first-order logic (FOL) symbol denoting a typed object (i.e., a FOL constant)."""

    name: str
    type_: str
    """Type of the denoted object."""
