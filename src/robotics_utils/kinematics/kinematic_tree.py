"""Represent the geometric state of an environment as a kinematic tree."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from robotics_utils.collision_models import CollisionModel
from robotics_utils.io import console
from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics.waypoints import Waypoints
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D
from robotics_utils.states import ContainerState, ObjectKinematicState

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.kinematics.configuration import Configuration


class KinematicTree:
    """A tree of coordinate frames specifying relative poses between entities."""

    def __init__(self, root_frame: str) -> None:
        """Initialize the kinematic tree's member variables based on its root frame."""
        self.root_frame = root_frame

        self.frames: dict[str, Pose3D] = {}
        """Maps the name of each frame to its relative pose."""

        self.children: defaultdict[str, set[str]] = defaultdict(set)
        """Maps the name of each frame to its set of child frames."""

        self.collision_models: dict[str, CollisionModel] = {}
        """Maps the name of each frame to its (optional) attached collision geometry."""

        # Record which frames correspond to objects or robot base poses
        self.object_names: set[str] = set()  # Object frame names: f"{object_name}"
        self.robot_names: set[str] = set()  # Base pose frame names: f"{robot_name}_base_pose"

        self.robot_configurations: dict[str, Configuration] = {}
        """Maps the name of each robot to its current joint configuration."""

        self.waypoints = Waypoints()  # Store navigation waypoints as 2D poses

        self.containers: dict[str, ContainerState] = {}
        """Maps the name of each container to its kinematic state."""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> KinematicTree:
        """Construct a KinematicTree instance using data from the given YAML file.

        :param yaml_path: YAML file containing data representing the kinematic state
        :return: Constructed KinematicTree instance
        """
        yaml_data: dict[str, Any] = load_yaml_data(
            yaml_path,
            required_keys={"robots", "objects"},
        )
        default_frame = yaml_data.get("default_frame", DEFAULT_FRAME)

        tree = KinematicTree(root_frame=default_frame)

        for robot_name, robot_data in yaml_data["robots"].items():
            base_pose_data = robot_data.get("base_pose")
            if base_pose_data is None:
                raise KeyError(f"Robot '{robot_name}' has no base pose specified.")
            base_pose = Pose3D.from_yaml_data(base_pose_data, default_frame=default_frame)

            tree.robot_names.add(robot_name)
            tree.set_robot_base_pose(robot_name, base_pose)
            tree.robot_configurations[robot_name] = {}  # Default: No robot configurations in YAML

        for obj_name, obj_data in yaml_data["objects"].items():
            tree.object_names.add(obj_name)

            pose_data = obj_data.get("pose")
            if pose_data is not None:
                obj_pose = Pose3D.from_yaml_data(pose_data, default_frame=default_frame)
                tree.set_object_pose(obj_name, obj_pose)

            collision_data = obj_data.get("collision_model")
            if collision_data is not None:
                c_model = CollisionModel.from_yaml_data(collision_data, yaml_path=yaml_path)
                tree.set_collision_model(obj_name, c_model)

            container_data = obj_data.get("container")
            if container_data is not None:
                container_state = ContainerState.from_yaml_data(
                    container_name=obj_name,
                    yaml_data=container_data,
                    yaml_path=yaml_path,
                    object_states=tree.known_object_states,
                )
                tree.add_container(container_state)

        # Import navigation waypoints from YAML if present
        if "waypoints" in yaml_data:
            tree.waypoints = Waypoints.from_yaml(yaml_path)

        return tree

    @property
    def known_object_poses(self) -> dict[str, Pose3D]:
        """Create and return a dictionary mapping object names to their 3D poses (if known)."""
        return {
            obj_name: self.frames[obj_name]
            for obj_name in self.object_names
            if obj_name in self.frames
        }

    @property
    def known_object_states(self) -> dict[str, ObjectKinematicState]:
        """Create and return a dictionary mapping object names to kinematic states (if known).

        :return: A map from object names to their kinematic states (if fully known)
        """
        object_states: dict[str, ObjectKinematicState] = {}
        for obj_name, obj_pose in self.known_object_poses.items():
            collision_model = self.get_collision_model(obj_name)
            if collision_model is None:
                continue

            object_states[obj_name] = ObjectKinematicState(obj_name, obj_pose, collision_model)

        return object_states

    @property
    def robot_base_poses(self) -> dict[str, Pose3D]:
        """Create and return a dictionary mapping robot names to their base poses."""
        return {r_name: self.frames[f"{r_name}_base_pose"] for r_name in self.robot_names}

    def _update_frame(self, frame_name: str, pose: Pose3D) -> None:
        """Update the named frame with the given relative pose.

        :param frame_name: Name of the reference frame added or updated
        :param pose: New relative pose of the frame
        """
        prev_parent_frame = self.get_parent_frame(frame_name)
        if prev_parent_frame is not None:  # Remove the frame from its previous parent's children
            self.children[prev_parent_frame].remove(frame_name)

        self.frames[frame_name] = pose
        self.children[pose.ref_frame].add(frame_name)  # Add this frame to its parent's children

    def valid_frame(self, frame_name: str) -> bool:
        """Evaluate whether the given frame name is valid within the kinematic tree."""
        return frame_name in self.frames or frame_name == self.root_frame

    def get_parent_frame(self, child_frame: str) -> str | None:
        """Retrieve the parent frame of the given child frame.

        :param child_frame: Frame whose parent frame is retrieved
        :return: Name of the reference frame of the child frame (None if parent frame is unknown)
        """
        child_pose = self.frames.get(child_frame)
        return None if child_pose is None else child_pose.ref_frame

    def set_object_pose(self, obj_name: str, new_pose: Pose3D) -> None:
        """Set the pose of the named object.

        :param obj_name: Name of the object assigned the given pose
        :param new_pose: New 3D pose of the object
        :raises ValueError: If the object name is unrecognized
        """
        if obj_name not in self.object_names:
            raise ValueError(f"Cannot set the pose of an unknown object: '{obj_name}'.")

        self._update_frame(obj_name, new_pose)

    def get_object_pose(self, obj_name: str) -> Pose3D:
        """Retrieve the pose of the named object.

        :param obj_name: Name of an object in the kinematic state
        :return: Pose of the object
        :raises ValueError: If the object name is unrecognized
        :raises KeyError: If the requested object pose is unknown
        """
        if obj_name not in self.object_names:
            raise ValueError(f"Cannot get the pose of an unknown object: '{obj_name}'.")

        if obj_name not in self.frames:
            raise KeyError(f"The pose of object '{obj_name}' is unknown.")

        return self.frames[obj_name]

    def set_robot_base_pose(self, robot_name: str, new_pose: Pose3D) -> None:
        """Set the base pose of the named robot.

        :param robot_name: Name of the robot assigned the given base pose
        :param new_pose: New base pose of the robot
        :raises ValueError: If the robot name is unrecognized
        """
        if robot_name not in self.robot_names:
            raise ValueError(f"Cannot get the base pose of an unknown robot: '{robot_name}'.")

        self._update_frame(f"{robot_name}_base_pose", new_pose)

    def get_robot_base_pose(self, robot_name: str) -> Pose3D:
        """Retrieve the base pose of the named robot.

        :param robot_name: Name of a robot
        :return: Base pose of the robot
        :raises ValueError: If the robot name is unrecognized
        :raises KeyError: If the requested robot base pose is unknown
        """
        if robot_name not in self.robot_names:
            raise ValueError(f"Cannot get the base pose of an unknown robot: '{robot_name}'.")

        base_pose_frame = f"{robot_name}_base_pose"
        if base_pose_frame not in self.frames:
            raise KeyError(f"The base pose of robot '{robot_name}' is unknown.")

        return self.frames[base_pose_frame]

    def set_collision_model(self, frame_name: str, collision_model: CollisionModel) -> None:
        """Set the collision geometry attached to the named frame.

        :param frame_name: Name of the frame to which the collision model is attached
        :param collision_model: Rigid-body collision geometry (primitive shape(s) and/or mesh(es))
        """
        if not self.valid_frame(frame_name):
            console.print(f"[yellow]Invalid frame '{frame_name}' had its collision model set.[/]")

        self.collision_models[frame_name] = collision_model

    def get_collision_model(self, frame_name: str) -> CollisionModel | None:
        """Retrieve the collision model attached to the named frame.

        :param frame_name: Name of the frame of the returned collision geometry
        :return: Collision model for the frame, or None if the frame has no attached geometry
        """
        return self.collision_models.get(frame_name)

    def add_container(self, container_state: ContainerState) -> None:
        """Update the state of the kinematic tree for the given container."""
        self.containers[container_state.name] = container_state
        container_state.update_kinematic_tree(self)

    def open_container(self, container_name: str) -> None:
        """Open the named container and update the state accordingly."""
        if container_name not in self.containers:
            raise KeyError(f"Cannot open unknown container: '{container_name}'.")
        self.containers[container_name].open(self)

    def close_container(self, container_name: str) -> None:
        """Close the named container and update the state accordingly."""
        if container_name not in self.containers:
            raise KeyError(f"Cannot close unknown container: '{container_name}'.")
        self.containers[container_name].close(self)

    def remove_object(self, obj_name: str) -> ObjectKinematicState | None:
        """Remove the named object from the kinematic state.

        :param obj_name: Name of the object removed from the state
        :return: Kinematic state of the object before it was removed (None if unknown)
        :raises KeyError: If the object doesn't exist in the kinematic tree
        :raises ValueError: If the frame of the object has child frames
        """
        if obj_name not in self.object_names:
            raise KeyError(f"Cannot remove unknown object '{obj_name}' from the kinematic tree.")

        if self.children[obj_name]:
            raise ValueError(
                f"Cannot remove object '{obj_name}' from the kinematic state "
                f"because it has child frames: {self.children[obj_name]}.",
            )

        self.object_names.remove(obj_name)
        self.children.pop(obj_name)

        parent_frame = self.get_parent_frame(obj_name)
        if parent_frame is not None:
            self.children[parent_frame].remove(obj_name)

        # Attempt to clear the object's pose and collision model, if they exist
        removed_pose = self.frames.pop(obj_name, None)
        removed_collision_model = self.collision_models.pop(obj_name, None)

        if removed_pose is None or removed_collision_model is None:
            return None

        return ObjectKinematicState(obj_name, removed_pose, removed_collision_model)
