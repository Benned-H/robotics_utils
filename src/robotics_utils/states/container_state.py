"""Define a class to represent the state of an openable/closable container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from robotics_utils.collision_models import CollisionModel
from robotics_utils.io.logging import log_info

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.kinematics import KinematicTree
    from robotics_utils.states.object_kinematic_state import ObjectKinematicState

ContainerStatus = Literal["open", "closed"]
"""Status describing whether a physical container is open or closed."""


@dataclass
class ContainerState:
    """The kinematic state of a physical container in the environment."""

    name: str
    status: ContainerStatus
    open_model: CollisionModel
    closed_model: CollisionModel

    contained_objects: dict[str, ObjectKinematicState]
    """A map from contained objects' names to their kinematic states."""

    @classmethod
    def from_yaml_data(
        cls,
        container_name: str,
        yaml_data: dict[str, Any],
        yaml_path: Path,
        object_states: dict[str, ObjectKinematicState],
    ) -> ContainerState:
        """Construct a ContainerState instance from a dictionary of YAML data.

        :param container_name: Name of the container
        :param yaml_data: YAML data specifying the initial state of the container
        :param yaml_path: Path to the YAML file the YAML data was loaded from
        :param object_states: Map from names of objects to their kinematic states
        :return: Constructed initial state of the container
        :raises KeyError: If a contained object's kinematic state is not provided
        """
        for required_key in ["status", "open_model", "closed_model"]:
            if required_key not in yaml_data:
                raise KeyError(f"ContainerState needs YAML key '{required_key}', got {yaml_data}")

        status = yaml_data["status"]
        if status not in get_args(ContainerStatus):
            raise ValueError(f"Container must be 'open' or 'closed'; got '{status}'.")

        open_model_data = yaml_data["open_model"]
        open_model = CollisionModel.from_yaml_data(open_model_data, yaml_path)

        closed_model_data = yaml_data["closed_model"]
        closed_model = CollisionModel.from_yaml_data(closed_model_data, yaml_path)

        contained_objects: dict[str, ObjectKinematicState] = {}
        contained_object_names: list[str] = yaml_data.get("contains", [])
        for obj_name in contained_object_names:
            if obj_name not in object_states:
                raise KeyError(
                    f"The kinematic state of object '{obj_name}' is needed to "
                    f"construct the container '{container_name}'.",
                )

            contained_objects[obj_name] = object_states[obj_name]

        return ContainerState(container_name, status, open_model, closed_model, contained_objects)

    @property
    def is_open(self) -> bool:
        """Check whether the container is open."""
        return self.status == "open"

    @property
    def is_closed(self) -> bool:
        """Check whether the container is closed."""
        return self.status == "closed"

    @property
    def current_collision_model(self) -> CollisionModel:
        """Retrieve the current collision model of the container."""
        return self.open_model if self.is_open else self.closed_model

    def update_kinematic_tree(self, tree: KinematicTree) -> None:
        """Update the given kinematic tree according to the container state."""
        tree.set_collision_model(self.name, self.current_collision_model)

        if self.status == "open":  # If open, update all contained objects' state
            for obj_name, obj_state in self.contained_objects.items():
                if obj_name not in tree.object_names:  # Add the object if it's new to the tree
                    tree.object_names.add(obj_name)

                tree.set_object_pose(obj_name, obj_state.pose)
                tree.set_collision_model(obj_name, obj_state.collision_model)

        elif self.status == "closed":  # If closed, clear contained objects' state in the tree
            for obj_name in self.contained_objects:
                removed_obj_state = tree.remove_object(obj_name)
                if removed_obj_state is None:
                    raise RuntimeError(f"Could not find the state of object '{obj_name}'.")

                self.contained_objects[obj_name] = removed_obj_state

    def open(self, tree: KinematicTree) -> None:
        """Open the container and update the kinematic tree accordingly."""
        if self.status == "open":
            log_info(f"Container '{self.name}' is already open!")
            return

        self.status = "open"
        self.update_kinematic_tree(tree)

    def close(self, tree: KinematicTree) -> None:
        """Close the container and update the kinematic tree accordingly."""
        if self.status == "closed":
            log_info(f"Container '{self.name}' is already closed!")
            return

        self.status = "closed"
        self.update_kinematic_tree(tree)
