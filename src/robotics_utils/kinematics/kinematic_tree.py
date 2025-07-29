"""Represent the geometric state of an environment as a kinematic tree."""

from __future__ import annotations

from pathlib import Path

from robotics_utils.filesystem.yaml_utils import (
    load_collision_models,
    load_named_poses,
    load_object_types,
)
from robotics_utils.kinematics import Configuration
from robotics_utils.kinematics.collision_models import CollisionModel, MeshSimplifier
from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.kinematics.waypoints import Waypoints


class KinematicTree:
    """A tree of coordinate frames specifying relative poses between entities."""

    def __init__(self) -> None:
        """Initialize the kinematic tree's member variables as empty."""
        self.frames: dict[str, Pose3D] = {}  # Map each frame name to its relative pose
        self.children: dict[str, set[str]] = {}  # Map each frame name to its child frames

        # Map each frame name to its (optional) attached collision geometry
        self.collision_models: dict[str, CollisionModel] = {}

        # Record which frames correspond to objects or robot base poses
        self.object_names: set[str] = set()  # Object frame names: f"{object_name}"
        self.robot_names: set[str] = set()  # Base pose frame names: f"{robot_name}_base_pose"

        self.object_types: dict[str, set[str]] = {}  # Map each object's name to its type(s)

        # Store robot configurations to represent actuated joints in the kinematic tree
        self.robot_configurations: dict[str, Configuration] = {}  # Map robot names to configs

        self.waypoints = Waypoints()  # Store navigation waypoints as 2D poses

    @classmethod
    def from_yaml(cls, yaml_path: Path, simplifier: MeshSimplifier) -> KinematicTree:
        """Construct a KinematicTree instance using data from the given YAML file.

        :param yaml_path: YAML file containing data representing the kinematic state
        :param simplifier: Used to simplify any imported collision meshes
        :return: Constructed KinematicTree instance
        """
        tree = KinematicTree()

        for obj_name, obj_pose in load_named_poses(yaml_path, "object_poses").items():
            tree.set_object_pose(obj_name, obj_pose)

        for robot_name, base_pose in load_named_poses(yaml_path, "robot_base_poses").items():
            tree.set_robot_base_pose(robot_name, base_pose)
            tree.robot_configurations[robot_name] = {}  # Default: No robot configurations in YAML

        tree.waypoints = Waypoints.from_yaml(yaml_path)
        tree.collision_models = load_collision_models(yaml_path, simplifier)
        tree.object_types = load_object_types(yaml_path)

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

        # Initialize the children sets for this frame and (if necessary) its parent frame
        self.children[frame_name] = self.children.get(frame_name, set())
        self.children[pose.ref_frame] = self.children.get(pose.ref_frame, set())
        self.children[pose.ref_frame].add(frame_name)  # Add this frame to its parent's children

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
        if frame_name not in self.frames:
            raise KeyError(f"Cannot set collision model for unknown frame: '{frame_name}'.")

        self.collision_models[frame_name] = collision_model

    def get_collision_model(self, frame_name: str) -> CollisionModel | None:
        """Retrieve the collision geometry attached to the named frame (None if no geometry)."""
        return self.collision_models.get(frame_name)

    def add_object_type(self, obj_name: str, obj_type: str) -> None:
        """Add an object type for the named object.

        :param obj_name: Name of an object in the kinematic state
        :param obj_type: Object type added to the object's types
        :raises KeyError: If an invalid object name is given
        """
        if obj_name not in self.object_names:
            raise KeyError(f"Cannot add object type for unknown object: '{obj_name}'.")
        self.object_types[obj_name] = self.object_types.get(obj_name, set())  # Initialize type set
        self.object_types[obj_name].add(obj_type)

    def get_object_types(self, obj_name: str) -> set[str]:
        """Retrieve the object types of the named object."""
        if obj_name not in self.object_types:
            raise KeyError(f"Cannot get types of unknown object: '{obj_name}'.")
        return self.object_types[obj_name]
