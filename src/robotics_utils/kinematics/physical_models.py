"""Define classes to model physical objects and containers in the environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from robotics_utils.collision_models import CollisionModel
from robotics_utils.kinematics import Pose3D


class ContainerState(Enum):
    """An enumeration of possible states of a physical container."""

    CLOSED = 0
    OPEN = 1


@dataclass(frozen=True)
class ObjectModel:
    """A physical object in the environment."""

    name: str
    pose: Pose3D
    collision_model: CollisionModel


class Container:
    """A physical container in the environment, potentially containing objects."""

    def __init__(
        self,
        closed_model: CollisionModel,
        open_model: CollisionModel,
        initial_state: ContainerState,
        contained_objects: set[ObjectModel],
    ) -> None:
        """Initialize the container's collision models, initial state, and contained objects."""
        self._closed_model = closed_model
        self._open_model = open_model

        self.state = initial_state
        self.contained_objects = contained_objects

    @property
    def collision_model(self) -> CollisionModel:
        """Retrieve the current collision model of the container."""
        return self._closed_model if self.state == ContainerState.CLOSED else self._open_model
