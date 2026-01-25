"""Define a class to represent an object-centric environment state."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from robotics_utils.collision_models import CollisionModel
from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D
from robotics_utils.states.container_state import ContainerState
from robotics_utils.states.kinematic_tree import KinematicTree
from robotics_utils.states.object_states import ObjectKinematicState

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.kinematics import Configuration


class PoseSource(Enum):
    """An enumeration of possible sources for object poses."""

    KNOWN = 1
    ESTIMATED = 2


class ObjectCentricState:
    """An object-centric state of an environment."""

    def __init__(self, robot_names: list[str], object_names: list[str], root_frame: str) -> None:
        """Initialize the object-centric state with lists of known robots and objects.

        :param robot_names: Names of robots in the environment
        :param object_names: Names of objects in the environment
        :param root_frame: Root frame used for the kinematic tree
        """
        self._robot_names = robot_names
        self._object_names = object_names

        self.kinematic_tree = KinematicTree(root_frame=root_frame)
        self._pose_sources: dict[str, PoseSource] = {}
        """A map from frame names to the source of their stored poses."""

        self.robot_configurations: dict[str, Configuration] = {}
        """A map from the name of each robot to its current joint configuration."""

        self._containers: dict[str, ContainerState] = {}
        """A map from the name of each container to its current state."""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ObjectCentricState:
        """Construct an ObjectCentricState instance using data from the given YAML file."""
        expected_keys = {"robots", "objects"}
        yaml_data: dict[str, Any] = load_yaml_data(yaml_path, required_keys=expected_keys)

        default_frame = yaml_data.get("default_frame", DEFAULT_FRAME)

        robot_names = list(yaml_data["robots"].keys())
        object_names = list(yaml_data["objects"].keys())

        state = ObjectCentricState(
            robot_names=robot_names,
            object_names=object_names,
            root_frame=default_frame,
        )

        for robot_name, robot_data in yaml_data["robots"].items():
            base_pose_data = robot_data.get("base_pose")
            if base_pose_data is None:
                raise KeyError(f"Robot '{robot_name}' has no base pose specified.")
            base_pose = Pose3D.from_yaml_data(base_pose_data, default_frame=default_frame)

            state.set_robot_base_pose(robot_name, base_pose)

        for obj_name, obj_data in yaml_data["objects"].items():
            pose_data = obj_data.get("pose")
            if pose_data is not None:
                obj_pose = Pose3D.from_yaml_data(pose_data, default_frame=default_frame)
                state.set_known_object_pose(obj_name, obj_pose)

            collision_data = obj_data.get("collision_model")
            if collision_data is not None:
                c_model = CollisionModel.from_yaml_data(collision_data, yaml_path=yaml_path)
                state.kinematic_tree.set_collision_model(obj_name, c_model)

            container_data = obj_data.get("container")
            if container_data is not None:
                container_state = ContainerState.from_yaml_data(
                    container_name=obj_name,
                    yaml_data=container_data,
                    yaml_path=yaml_path,
                    object_states=state.available_kinematic_states,
                )
                state.add_container(container_state)

        return state

    @property
    def robot_names(self) -> list[str]:
        """Provide read-only access to the known robot names in the object-centric state."""
        return self._robot_names

    @property
    def object_names(self) -> list[str]:
        """Provide read-only access to the known object names in the object-centric state."""
        return self._object_names

    @property
    def object_poses(self) -> dict[str, Pose3D]:
        """Retrieve a dictionary mapping object names to their poses (if available)."""
        return self.kinematic_tree.get_poses(frame_names=set(self._object_names))

    @property
    def robot_base_poses(self) -> dict[str, Pose3D]:
        """Retrieve a dictionary mapping robot names to their base poses (if available)."""
        frame_names = {f"{robot_name}_base_pose" for robot_name in self._robot_names}
        return self.kinematic_tree.get_poses(frame_names=frame_names)

    @property
    def available_kinematic_states(self) -> dict[str, ObjectKinematicState]:
        """Retrieve a dictionary mapping object names to their kinematic states (if available)."""
        obj_states: dict[str, ObjectKinematicState] = {}
        for obj_name, obj_pose in self.object_poses.items():
            collision_model = self.kinematic_tree.get_collision_model(frame_name=obj_name)
            if collision_model is not None:
                obj_states[obj_name] = ObjectKinematicState(obj_name, obj_pose, collision_model)

        return obj_states

    def add_object(self, obj_name: str) -> None:
        """Add the given object name to the set of known object names."""
        if obj_name not in self._object_names:
            self._object_names.append(obj_name)

    def add_robot(self, robot_name: str) -> None:
        """Add the given robot name to the set of known robot names."""
        if robot_name not in self._robot_names:
            self._robot_names.append(robot_name)

    def set_known_object_pose(self, obj_name: str, pose: Pose3D) -> None:
        """Set the pose of the named object from a "known" source of truth.

        :param obj_name: Name of the object assigned the pose
        :param pose: Updated object pose from a "known" authoritative source
        :raises ValueError: If the object name is unrecognized
        """
        if obj_name not in self._object_names:
            raise ValueError(f"Cannot set the pose of an unknown object: '{obj_name}'.")

        self.kinematic_tree.set_pose(obj_name, pose)
        self._pose_sources[obj_name] = PoseSource.KNOWN

    def set_estimated_object_pose(self, obj_name: str, pose: Pose3D) -> None:
        """Set the pose of the named object from an estimated source.

        :param obj_name: Name of the object assigned the pose
        :param pose: Updated object pose from an estimated source
        :raises ValueError: If the object name is unrecognized
        """
        if obj_name not in self._object_names:
            raise ValueError(f"Cannot set the pose of an unknown object: '{obj_name}'.")

        # Don't allow estimated poses to overwrite known poses
        if self._pose_sources.get(obj_name) == PoseSource.KNOWN:
            return

        self.kinematic_tree.set_pose(obj_name, pose)
        self._pose_sources[obj_name] = PoseSource.ESTIMATED

    def get_object_pose(self, obj_name: str) -> Pose3D | None:
        """Retrieve the pose of the named object.

        :param obj_name: Name of an object in the environment
        :return: Current pose of the object
        :raises ValueError: If the object name is unrecognized
        """
        if obj_name not in self._object_names:
            raise ValueError(f"Cannot get the pose of an unknown object: '{obj_name}'.")

        pose_map = self.kinematic_tree.get_poses(frame_names={obj_name})
        return pose_map.get(obj_name)

    def clear_object_pose(self, obj_name: str) -> Pose3D | None:
        """Clear the pose of the named object while keeping it registered in the state.

        :param obj_name: Name of the object whose pose is cleared
        :return: Object's pose before it was cleared, or None if the pose was unknown
        :raises ValueError: If the object name is unrecognized
        """
        if obj_name not in self._object_names:
            raise ValueError(f"Cannot clear the pose of an unknown object: '{obj_name}'.")

        self._pose_sources.pop(obj_name, None)
        return self.kinematic_tree.clear_pose(frame_name=obj_name)

    def set_robot_base_pose(self, robot_name: str, base_pose: Pose3D) -> None:
        """Set the base pose of the named robot.

        :param robot_name: Name of the robot assigned the given base pose
        :param base_pose: New base pose of the robot
        :raises ValueError: If the robot name is unrecognized
        """
        if robot_name not in self._robot_names:
            raise ValueError(f"Cannot set the base pose of an unknown robot: '{robot_name}'.")

        self.kinematic_tree.set_pose(f"{robot_name}_base_pose", base_pose)

    def get_robot_base_pose(self, robot_name: str) -> Pose3D | None:
        """Retrieve the base pose of the named robot.

        :param robot_name: Name of a robot
        :return: Base pose of the robot, if known, else None
        :raises ValueError: If the robot name is unrecognized
        """
        if robot_name not in self._robot_names:
            raise ValueError(f"Cannot get the base pose of an unknown robot: '{robot_name}'.")

        base_pose_frame = f"{robot_name}_base_pose"
        pose_map = self.kinematic_tree.get_poses(frame_names={base_pose_frame})
        return pose_map.get(base_pose_frame)

    def add_container(self, container_state: ContainerState) -> None:
        """Update the object-centric state based on the given container state."""
        self._containers[container_state.name] = container_state
        container_state.update_state(self)

    def open_container(self, container_name: str) -> None:
        """Open the named container and update the state accordingly.

        :raises ValueError: If the container name is unrecognized
        """
        if container_name not in self._containers:
            raise ValueError(f"Cannot open unknown container: '{container_name}'.")
        self._containers[container_name].open(self)

    def close_container(self, container_name: str) -> None:
        """Close the named container and update the state accordingly.

        :raises ValueError: If the container name is unrecognized
        """
        if container_name not in self._containers:
            raise ValueError(f"Cannot close unknown container: '{container_name}'.")
        self._containers[container_name].close(self)

    def update_estimated_poses(self, pose_estimates: dict[str, Pose3D]) -> None:
        """Update the object-centric state based on the given pose estimates.

        :param pose_estimates: Dictionary mapping frame names to their estimated poses
        """
        for obj_name in self.object_names:
            if self._pose_sources.get(obj_name) == PoseSource.KNOWN:
                continue  # Ignore objects that already have known poses

            estimate = pose_estimates.get(obj_name)
            if estimate is not None:
                self.set_estimated_object_pose(obj_name, estimate)
