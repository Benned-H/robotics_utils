"""Define a base class that makes it easy to create Python-typed object types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseObjectType:
    """A base class for simple object types."""

    name: str
    visual_description: str | None = None

    def __str__(self) -> str:
        """Return a readable string representation of the object."""
        return f"{type(self).__name__}({self.name})"
