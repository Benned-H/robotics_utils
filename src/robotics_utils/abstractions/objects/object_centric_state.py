"""Define a class to represent the state of an object-centric environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObjectCentricState:
    """The state of an object-centric environment."""

    objects: dict[str, object]
    """Map from object names ot object instances in the environment."""
