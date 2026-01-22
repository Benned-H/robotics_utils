"""Represent the geometric state of an environment as a kinematic tree."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from robotics_utils.collision_models import CollisionModel
from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D
from robotics_utils.states import ContainerState, ObjectKinematicState

if TYPE_CHECKING:
    from pathlib import Path


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

    def remove_object(self, obj_name: str) -> ObjectKinematicState | None:
        """Remove the named object from the kinematic state.

        :param obj_name: Name of the object removed from the state
        :return: Kinematic state of the object before it was removed (None if unknown)
        :raises ValueError: If the object doesn't exist in the kinematic tree
        :raises ValueError: If the frame of the object has child frames
        """
        if obj_name not in self.object_names:
            raise ValueError(f"Cannot remove unknown object '{obj_name}' from the kinematic tree.")

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
