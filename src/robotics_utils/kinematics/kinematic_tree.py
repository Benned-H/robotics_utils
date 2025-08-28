"""Represent the geometric state of an environment as a kinematic tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotics_utils.collision_models import CollisionModel
from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics.kinematics_core import DEFAULT_FRAME, Configuration
from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.kinematics.waypoints import Waypoints
from robotics_utils.world_models.containers import ContainerModel, ObjectModel

if TYPE_CHECKING:
    from pathlib import Path


class KinematicTree:
    """A tree of coordinate frames specifying relative poses between entities."""

    def __init__(self, root_frame: str) -> None:
        """Initialize the kinematic tree's member variables based on its root frame."""
        self.root_frame = root_frame

        self.frames: dict[str, Pose3D] = {}
        """Maps the name of each frame to its relative pose."""

        self.children: dict[str, set[str]] = {root_frame: set()}
        """Maps the name of each frame to its set of child frames."""

        self.collision_models: dict[str, CollisionModel] = {}
        """Maps the name of each frame to its (optional) attached collision geometry."""

        # Record which frames correspond to objects or robot base poses
        self.object_names: set[str] = set()  # Object frame names: f"{object_name}"
        self.robot_names: set[str] = set()  # Base pose frame names: f"{robot_name}_base_pose"

        self.robot_configurations: dict[str, Configuration] = {}
        """Maps the name of each robot to its current joint configuration."""

        self.waypoints = Waypoints()  # Store navigation waypoints as 2D poses

        self.container_models: dict[str, ContainerModel] = {}
        """Maps the name of each container to its kinematic model."""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> KinematicTree:
        """Construct a KinematicTree instance using data from the given YAML file.

        :param yaml_path: YAML file containing data representing the kinematic state
        :return: Constructed KinematicTree instance
        """
        full_yaml_data: dict[str, Any] = load_yaml_data(
            yaml_path,
            required_keys={"object_poses", "robot_base_poses"},
        )
        default_frame = full_yaml_data.get("default_frame", DEFAULT_FRAME)

        tree = KinematicTree(root_frame=default_frame)

        for obj_name, obj_pose in Pose3D.load_named_poses(yaml_path, "object_poses").items():
            tree.set_object_pose(obj_name, obj_pose)

        for robot_name, base_pose in Pose3D.load_named_poses(yaml_path, "robot_base_poses").items():
            tree.set_robot_base_pose(robot_name, base_pose)
            tree.robot_configurations[robot_name] = {}  # Default: No robot configurations in YAML

        tree.waypoints = Waypoints.from_yaml(yaml_path)

        # Initially, load all collision models into a temporary dictionary
        collision_models: dict[str, CollisionModel] = {}
        for m_name, m_data in full_yaml_data.get("collision_models", {}).items():
            collision_models[m_name] = CollisionModel.from_yaml_data(m_data, yaml_path)

        # Identify which collision models are used by containers
        containers_data = full_yaml_data.get("containers", {})
        used_by_containers: set[str] = set()  # Names of container collision models
        for c_data in containers_data.values():
            used_by_containers.add(c_data["closed_model"])
            used_by_containers.add(c_data["open_model"])

        # Add all collision models that aren't used as a container model
        for name, collision_model in collision_models.items():
            if name not in used_by_containers:
                tree.set_collision_model(frame_name=name, collision_model=collision_model)

        # Load any containers in the environment from YAML
        for c_name, c_data in containers_data.values():
            c = ContainerModel.from_yaml_data(c_name, c_data, collision_models, tree.object_poses)
            tree.add_container(c)

        return tree

    @property
    def object_poses(self) -> dict[str, Pose3D]:
        """Create and return a dictionary mapping object names to their 3D poses."""
        return {obj_name: self.frames[obj_name] for obj_name in self.object_names}

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

        # Ensure that this frame and its parent frame have children sets initialized
        self.children[frame_name] = self.children.get(frame_name, set())
        self.children[pose.ref_frame] = self.children.get(pose.ref_frame, set())
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
        """
        self._update_frame(obj_name, new_pose)
        self.object_names.add(obj_name)

    def get_object_pose(self, obj_name: str) -> Pose3D:
        """Retrieve the pose of the named object.

        :param obj_name: Name of an object in the kinematic state
        :return: Pose of the object
        :raises KeyError: If an invalid object name is given
        """
        if obj_name not in self.object_names or obj_name not in self.frames:
            raise KeyError(f"Cannot get pose of unknown object: '{obj_name}'.")

        return self.frames[obj_name]

    def set_robot_base_pose(self, robot_name: str, new_pose: Pose3D) -> None:
        """Set the base pose of the named robot.

        :param robot_name: Name of the robot assigned the given base pose
        :param new_pose: New base pose of the robot
        """
        self._update_frame(f"{robot_name}_base_pose", new_pose)
        self.robot_names.add(robot_name)

    def get_robot_base_pose(self, robot_name: str) -> Pose3D:
        """Retrieve the base pose of the named robot.

        :param robot_name: Name of a robot
        :return: Base pose of the robot
        :raises KeyError: If an invalid robot name is given
        """
        if robot_name not in self.robot_names or f"{robot_name}_base_pose" not in self.frames:
            raise KeyError(f"Cannot get base pose of unknown robot: '{robot_name}'.")

        return self.frames[f"{robot_name}_base_pose"]

    def set_collision_model(self, frame_name: str, collision_model: CollisionModel) -> None:
        """Set the collision geometry attached to the named frame.

        :param frame_name: Name of the frame to which the collision model is attached
        :param collision_model: Rigid-body collision geometry (primitive shape(s) and/or mesh(es))
        :raises KeyError: If an invalid frame name is given
        """
        if not self.valid_frame(frame_name):
            raise KeyError(f"Cannot set collision model for unknown frame: '{frame_name}'.")

        self.collision_models[frame_name] = collision_model

    def get_collision_model(self, frame_name: str) -> CollisionModel | None:
        """Retrieve the collision model attached to the named frame.

        :param frame_name: Name of the frame of the returned collision geometry
        :return: Collision model for the frame, or None if the frame has no attached geometry
        """
        if not self.valid_frame(frame_name):
            raise KeyError(f"Cannot get collision model for unknown frame: '{frame_name}'.")

        return self.collision_models.get(frame_name)

    def add_container(self, container: ContainerModel) -> None:
        """Add a container model to the kinematic tree and update the state accordingly."""
        self.container_models[container.name] = container
        container.update_kinematic_tree(self)

    def open_container(self, container_name: str) -> None:
        """Open the named container and update the state accordingly."""
        if container_name not in self.container_models:
            raise KeyError(f"Cannot open unknown container: '{container_name}'.")
        self.container_models[container_name].open(self)

    def close_container(self, container_name: str) -> None:
        """Close the named container and update the state accordingly."""
        if container_name not in self.container_models:
            raise KeyError(f"Cannot close unknown container: '{container_name}'.")
        self.container_models[container_name].close(self)

    def remove_object(self, obj_name: str) -> ObjectModel | None:
        """Remove the named object from the kinematic state.

        :param obj_name: Name of the object removed from the state
        :return: Model of the object's state before it was removed (None if no state existed)
        """
        if obj_name not in self.object_names:
            raise KeyError(f"Cannot remove unknown object '{obj_name}' from the kinematic tree.")
        self.object_names.remove(obj_name)

        if self.children[obj_name]:
            raise ValueError(
                f"Cannot remove object '{obj_name}' from the kinematic state "
                f"because it has child frames: {self.children[obj_name]}.",
            )
        self.children.pop(obj_name)

        parent_frame = self.get_parent_frame(obj_name)
        if parent_frame is not None:
            self.children[parent_frame].remove(obj_name)

        # Attempt to clear the object's pose and collision model, if they exist
        removed_pose = self.frames.pop(obj_name, None)
        removed_collision_model = self.collision_models.pop(obj_name, None)

        if removed_pose is None or removed_collision_model is None:
            return None

        return ObjectModel(obj_name, removed_pose, removed_collision_model)
