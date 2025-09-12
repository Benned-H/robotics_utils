"""Define a class to synchronize the MoveIt planning scene with an external kinematic state."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

import rospy
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import CollisionObject

from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.ros.msg_conversion import make_collision_object_msg, pose_from_msg, pose_to_msg
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.world_models.simulators import ObjectModel, Simulator

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.collision_models.collision_model import CollisionModel
    from robotics_utils.kinematics.poses import Pose3D
    from robotics_utils.motion_planning import MotionPlanningQuery
    from robotics_utils.robots.manipulator import Manipulator


class PlanningSceneManager(Simulator):
    """A manager to update the state of the MoveIt planning scene."""

    def __init__(self, body_frame: str = "body") -> None:
        """Initialize the manager's interface with the MoveIt planning scene."""
        self.planning_scene = PlanningSceneInterface()
        rospy.sleep(3)  # Allow time for the scene to initialize

        self.body_frame = body_frame

        self._added_objects: set[str] = set()
        """Names of all objects added to the planning scene (doesn't count hidden objects)."""

        self._hidden_objects: dict[str, CollisionObject] = {}
        """Hidden objects are ignored for the purposes of collision checking."""

        self._attached_objects: dict[str, set[str]] = defaultdict(set)
        """Maps each robot name to the names of objects attached to that robot."""

    @classmethod
    def populate_from_yaml(cls, yaml_path: Path) -> tuple[bool, str]:
        """Populate the MoveIt planning scene by loading from the given YAML file."""
        tree = KinematicTree.from_yaml(yaml_path)
        scene = PlanningSceneManager()
        success = scene.synchronize_state(tree)
        message = (
            f"Loaded MoveIt planning scene from YAML file: {yaml_path}."
            if success
            else f"Could not fully load planning scene from the YAML file: {yaml_path}."
        )
        return success, message

    def add_object(self, obj_model: ObjectModel) -> bool:
        """Add an object to the MoveIt planning scene.

        :param obj_model: Geometric model of the object to be added
        :return: True if the object was successfully added, else False
        """
        TransformManager.broadcast_transform(obj_model.name, obj_model.pose)
        pose_b_o = TransformManager.convert_to_frame(obj_model.pose, self.body_frame)

        collision_obj_msg = make_collision_object_msg(obj_model)
        collision_obj_msg.pose = pose_to_msg(pose_b_o)  # Ensure that the pose is in body frame

        return self.add_object_msg(collision_obj_msg)

    def add_object_msg(self, collision_obj_msg: CollisionObject) -> bool:
        """Add a moveit_msgs/CollisionObject message to the MoveIt planning scene.

        :param collision_obj_msg: ROS message representing an object's collision geometry
        :return: True if the object was successfully added, else False
        """
        self.planning_scene.add_object(collision_obj_msg)

        object_exists = self.wait_until_object_exists(collision_obj_msg.id)
        if not object_exists:
            rospy.logerr(f"Failed to add '{collision_obj_msg.id}' to the MoveIt planning scene.")

        if object_exists:
            self._added_objects.add(collision_obj_msg.id)

        return object_exists

    def remove_object(self, obj_name: str) -> bool:
        """Remove the named object from the MoveIt planning scene.

        :param obj_name: Name of the object to be removed
        :return: True if the object was successfully removed, else False
        """
        self.planning_scene.remove_world_object(obj_name)
        removed = self.wait_until_object_removed(obj_name)

        if removed:
            self._added_objects.remove(obj_name)

        return removed

    def hide_object(self, obj_name: str) -> bool:
        """Hide the named object for the purposes of collision checking.

        :param obj_name: Object to be hidden (i.e., ignored during collision checking)
        :return: True if the object was successfully hidden, else False
        """
        if obj_name in self._hidden_objects:
            rospy.logwarn(f"Object '{obj_name}' is already hidden from the planning scene.")
            return False

        self._hidden_objects[obj_name] = self.get_object_msg(obj_name)
        object_hidden = self.remove_object(obj_name)
        return object_hidden and obj_name in self._hidden_objects

    def unhide_object(self, obj_name: str) -> bool:
        """Unhide the named object for the purposes of collision checking.

        :param obj_name: Name of the object to be added back for collision checking
        :return: True if the object was successfully unhidden, else False
        """
        object_msg = self._hidden_objects.pop(obj_name, None)
        if object_msg is None:
            rospy.logwarn(f"Cannot unhide object '{obj_name}' because it isn't hidden.")
            return False

        return self.add_object_msg(object_msg)

    def hide_all_objects(self) -> bool:
        """Hide all objects in the planning scene for the purposes of collision checking.

        :return: True if all objects were successfully hidden, else False
        """
        objects_to_hide = self._added_objects.copy()
        all_hidden = True  # Have all objects succeeded so far?
        for obj_name in objects_to_hide:
            hidden = self.hide_object(obj_name)
            all_hidden = all_hidden and hidden

        return all_hidden

    def unhide_all_objects(self) -> bool:
        """Unhide all objects for the purposes of collision checking.

        :return: True if all objects were successfully unhidden, else False
        """
        objects_to_unhide = self._hidden_objects.copy()
        all_unhidden = True  # Have all objects succeeded so far?
        for obj_name in objects_to_unhide:
            unhidden = self.unhide_object(obj_name)
            all_unhidden = all_unhidden and unhidden

        return all_unhidden

    def apply_query_ignores(self, query: MotionPlanningQuery) -> bool:
        """Hide objects as indicated by the given motion planning query."""
        if query.ignore_all_collisions:
            return self.hide_all_objects()

        if query.ignored_objects:
            all_hidden = True
            for obj_to_hide in query.ignored_objects:
                hidden = self.hide_object(obj_to_hide)
                all_hidden = all_hidden and hidden
            return all_hidden

        return True

    def unapply_query_ignores(self, query: MotionPlanningQuery) -> bool:
        """Unhide objects as indicated by the given motion planning query."""
        if query.ignore_all_collisions:
            return self.unhide_all_objects()

        if query.ignored_objects:
            all_unhidden = True
            for obj_to_unhide in query.ignored_objects:
                unhidden = self.unhide_object(obj_to_unhide)
                all_unhidden = all_unhidden and unhidden
            return all_unhidden

        return True

    def get_attached_objects(self, robot_name: str) -> set[str]:
        """Retrieve the names of objects attached to the named robot (defaults to empty set)."""
        return self._attached_objects[robot_name]

    def get_object_msg(self, obj_name: str) -> CollisionObject:
        """Retrieve the CollisionObject message for the named object in the planning scene."""
        object_msgs: dict[str, CollisionObject] = self.planning_scene.get_objects([obj_name])
        return object_msgs[obj_name]

    def set_object_pose(self, obj_name: str, new_pose: Pose3D) -> None:
        """Update the pose of the named object in the MoveIt planning scene."""
        new_pose_b_o = TransformManager.convert_to_frame(new_pose, self.body_frame)

        move_object_msg = CollisionObject()
        move_object_msg.id = obj_name
        move_object_msg.operation = CollisionObject.MOVE
        move_object_msg.pose = pose_to_msg(new_pose_b_o)
        move_object_msg.header.frame_id = new_pose_b_o.ref_frame

        self.planning_scene.add_object(move_object_msg)

    def set_collision_model(self, obj_name: str, collision_model: CollisionModel) -> None:
        """Replace the collision geometry of the named object in the MoveIt planning scene."""
        obj_pose_msg = self.planning_scene.get_object_poses([obj_name])[obj_name]
        obj_pose = pose_from_msg(obj_pose_msg)
        object_model = ObjectModel(obj_name, obj_pose, collision_model)

        self.add_object(object_model)

    def synchronize_state(self, tree: KinematicTree, attempts_per_obj: int = 3) -> bool:
        """Update the MoveIt planning scene to reflect the given environment state."""
        all_added = True
        for object_model in tree.object_models.values():
            object_added = self.add_object(object_model)
            attempts_left = attempts_per_obj - 1

            while (attempts_left > 0) and not object_added:
                object_added = self.add_object(object_model)
                attempts_left -= 1

            all_added = all_added and object_added

        return all_added

    def wait_until_object_exists(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the MoveIt planning scene contains the named object.

        :param name: Name of the object to find in the planning scene
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object appears in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name in self.planning_scene.get_known_object_names():
                return True
            time.sleep(0.1)

        return False

    def wait_until_object_removed(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the named object is removed from the MoveIt planning scene.

        :param name: Name of the object to be removed from the planning scene
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is removed in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name not in self.planning_scene.get_known_object_names():
                return True
            time.sleep(0.1)

        return False

    def wait_until_object_attached(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the named object is attached in the MoveIt planning scene.

        :param name: Name of the object to check for attachment
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is attached in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name in self.planning_scene.get_attached_objects():
                return True
            time.sleep(0.1)

        return False

    def wait_until_object_detached(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the named object is detached in the MoveIt planning scene.

        :param name: Name of the object to check for detachment
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is detached in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name not in self.planning_scene.get_attached_objects():
                return True
            time.sleep(0.1)

        return False

    def grasp_object(self, object_name: str, robot_name: str, manipulator: Manipulator) -> bool:
        """Grasp the named object using the named robot's specified manipulator."""
        if manipulator.gripper is None:
            rospy.logerr(f"Cannot grasp without a gripper on manipulator '{manipulator.name}'.")
            return False

        ee_link = manipulator.ee_link_name
        gripper_links = manipulator.gripper.link_names

        self.planning_scene.attach_object(object_name, link=ee_link, touch_links=gripper_links)
        is_attached = self.wait_until_object_attached(object_name)

        if is_attached:
            self._attached_objects[robot_name].add(object_name)

        return is_attached

    def release_object(self, object_name: str, robot_name: str, manipulator: Manipulator) -> bool:
        """Release the named object using the named robot's specified manipulator."""
        if object_name not in self._attached_objects[robot_name]:
            rospy.logwarn(f"Cannot release unattached object '{object_name}' with '{robot_name}'.")
            return False

        self.planning_scene.remove_attached_object(link=manipulator.ee_link_name, name=object_name)
        is_detached = self.wait_until_object_detached(object_name)

        if is_detached:
            self._attached_objects[robot_name].remove(object_name)

        return is_detached
