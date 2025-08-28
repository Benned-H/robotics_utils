"""Define classes to represent closable/openable containers in the environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from robotics_utils.io.logging import log_info

if TYPE_CHECKING:
    from robotics_utils.collision_models import CollisionModel
    from robotics_utils.kinematics import Pose3D
    from robotics_utils.kinematics.kinematic_tree import KinematicTree


@dataclass(frozen=True)
class ObjectModel:
    """A physical object in the environment."""

    name: str
    pose: Pose3D
    collision_model: CollisionModel


class ContainerState(Enum):
    """An enumeration of possible states of a physical container."""

    CLOSED = 0
    OPEN = 1

    @classmethod
    def from_string(cls, value: str) -> ContainerState:
        """Construct a ContainerState from a string."""
        state = {
            "closed": ContainerState.CLOSED,
            "open": ContainerState.OPEN,
        }.get(value.lower().strip())

        if state is None:
            raise ValueError(f"Cannot construct ContainerState from string: '{value}'.")
        return state


@dataclass
class ContainerModel:
    """A physical container in the environment, potentially containing objects."""

    name: str
    state: ContainerState
    closed_model: CollisionModel
    open_model: CollisionModel

    contained_objects: dict[str, ObjectModel]
    """Map from contained objects' names to their object models."""

    @classmethod
    def from_yaml_data(
        cls,
        name: str,
        yaml_data: dict[str, Any],
        collision_models: dict[str, CollisionModel],
        object_poses: dict[str, Pose3D],
    ) -> ContainerModel:
        """Construct a ContainerModel instance from a dictionary of YAML data.

        :param name: Name of the container being modeled
        :param yaml_data: YAML data specifying the container model
        :param collision_models: Map from collision model names to previously loaded models
        :param object_poses: Map from object names to previously loaded object poses
        """
        for required_key in ["state", "closed_model", "open_model"]:
            if required_key not in yaml_data:
                raise KeyError(f"ContainerModel needs YAML key '{required_key}', got {yaml_data}")

        initial_state = ContainerState.from_string(yaml_data["state"])

        closed_model = collision_models.get(yaml_data["closed_model"])
        if closed_model is None:
            raise KeyError(f"Missing closed model when constructing container '{name}'.")

        open_model = collision_models.get(yaml_data["open_model"])
        if open_model is None:
            raise KeyError(f"Missing open model when constructing container '{name}'.")

        contained_objects: dict[str, ObjectModel] = {
            obj_name: ObjectModel(obj_name, object_poses[obj_name], collision_models[obj_name])
            for obj_name in yaml_data.get("contains", [])
        }

        return ContainerModel(name, initial_state, closed_model, open_model, contained_objects)

    @property
    def collision_model(self) -> CollisionModel:
        """Retrieve the current collision model of the container."""
        return self.closed_model if self.is_closed else self.open_model

    @property
    def is_closed(self) -> bool:
        """Check whether the container is closed."""
        return self.state == ContainerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check whether the container is open."""
        return self.state == ContainerState.OPEN

    def update_kinematic_tree(self, tree: KinematicTree) -> None:
        """Update the given kinematic tree according to this container's state.

        :param tree: Kinematic model of the environment updated to reflect the container state
        """
        tree.set_collision_model(frame_name=self.name, collision_model=self.collision_model)

        if self.state == ContainerState.OPEN:  # If open, update all contained objects' state
            for obj_name, obj_model in self.contained_objects.items():
                tree.set_object_pose(obj_name, obj_model.pose)
                tree.set_collision_model(obj_name, obj_model.collision_model)

        elif self.state == ContainerState.CLOSED:  # If closed, clear all contained objects' state
            for obj_name in self.contained_objects:
                removed_obj_model = tree.remove_object(obj_name)
                if removed_obj_model is not None:
                    self.contained_objects[obj_name] = removed_obj_model

    def open(self, tree: KinematicTree) -> None:
        """Open the container and update the given kinematic tree accordingly.

        :param tree: Specifies the kinematic state of the environment
        """
        if self.state == ContainerState.OPEN:
            log_info(f"Container '{self.name}' is already open so it cannot be opened again.")
            return

        self.state = ContainerState.OPEN
        self.update_kinematic_tree(tree)

    def close(self, tree: KinematicTree) -> None:
        """Close the container and update the kinematic tree accordingly.

        :param tree: Specifies the kinematic state of the environment
        """
        if self.state == ContainerState.CLOSED:
            log_info(f"Container '{self.name}' is already closed so it cannot be closed again.")
            return

        self.state = ContainerState.CLOSED
        self.update_kinematic_tree(tree)
