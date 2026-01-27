"""Define a class to represent the state of an openable/closable container."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

from robotics_utils.collision_models import CollisionModel
from robotics_utils.io.logging import log_info
from robotics_utils.spatial import Pose3D

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.states.object_centric_state import ObjectCentricState


ContainerStatus = Literal["open", "closed"]
"""Status describing whether a physical container is open or closed."""


class ContainerState:
    """The kinematic state of a physical container in the environment."""

    def __init__(
        self,
        name: str,
        status: ContainerStatus,
        open_model: CollisionModel,
        closed_model: CollisionModel,
    ) -> None:
        """Initialize the state of the container."""
        self.name = name
        self.status = status
        self.open_model = open_model
        self.closed_model = closed_model

        self._contained_object_names: set[str] = set()
        """Set of the names of objects contained by the container."""

        self._closed_poses: dict[str, Pose3D] = {}
        """A map from contained objects' names to their poses when the container is closed."""

        self._open_poses: dict[str, Pose3D] = {}
        """A map from contained objects' names to their poses when the container is open."""

    @classmethod
    def from_yaml_data(
        cls,
        name: str,
        yaml_data: dict[str, Any],
        yaml_path: Path,
    ) -> ContainerState:
        """Construct a ContainerState instance from a dictionary of YAML data.

        :param name: Name of the container
        :param yaml_data: YAML data specifying the initial state of the container
        :param yaml_path: Path to the YAML file the YAML data was loaded from
        :return: Constructed initial state of the container
        :raises KeyError: If the YAML data is missing a required key
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

        container = ContainerState(name, status, open_model, closed_model)

        for obj_name, obj_data in yaml_data.get("contains", {}).items():
            open_data = obj_data["pose_when_open"]
            closed_data = obj_data["pose_when_closed"]

            open_pose = Pose3D.from_yaml_data(open_data, default_frame=name)
            closed_pose = Pose3D.from_yaml_data(closed_data, default_frame=name)

            container.add_object(obj_name, pose_when_open=open_pose, pose_when_closed=closed_pose)

        return container

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

    def contains(self, obj_name: str) -> bool:
        """Check whether the named object is contained in the container."""
        return obj_name in self._contained_object_names

    def get_contained_object_pose(self, obj_name: str) -> Pose3D:
        """Retrieve the container-relative pose of the named object.

        :raises ValueError: If the requested object is not in the container
        """
        if obj_name not in self._contained_object_names:
            raise ValueError(f"Container '{self.name}' does not contain object '{obj_name}'.")

        return self._open_poses[obj_name] if self.is_open else self._closed_poses[obj_name]

    def add_object(self, obj_name: str, pose_when_open: Pose3D, pose_when_closed: Pose3D) -> None:
        """Add an object to the container's internal state.

        :param obj_name: Name of the contained object
        :param pose_when_open: Container-relative object pose when the container is open
        :param pose_when_closed: Container-relative object pose when the container is closed
        """
        self._contained_object_names.add(obj_name)
        self._open_poses[obj_name] = pose_when_open
        self._closed_poses[obj_name] = pose_when_closed

    def update_state(self, state: ObjectCentricState) -> None:
        """Update an object-centric environment state according to this container state."""
        state.kinematic_tree.set_collision_model(self.name, self.current_collision_model)

        known_object_names = state.object_names
        for obj_name in self._contained_object_names:
            if obj_name not in known_object_names:
                state.add_object(obj_name)

            object_pose = self.get_contained_object_pose(obj_name)
            state.set_known_object_pose(obj_name=obj_name, pose=object_pose)

    def open(self, state: ObjectCentricState) -> None:
        """Open the container and update the given object-centric state accordingly."""
        if self.status == "open":
            log_info(f"Container '{self.name}' is already open.")
            return

        self.status = "open"
        self.update_state(state=state)

    def close(self, state: ObjectCentricState) -> None:
        """Close the container and update the given object-centric state accordingly."""
        if self.status == "closed":
            log_info(f"Container '{self.name}' is already closed.")
            return

        self.status = "closed"
        self.update_state(state=state)
