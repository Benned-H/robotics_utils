"""Define a class to synchronize the MoveIt planning scene with an external kinematic state."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

import rospy
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import CollisionObject as CollisionObjectMsg

from robotics_utils.ros.msg_conversion import (
    pose_from_msg,
    pose_to_msg,
    primitive_shape_to_msg,
    trimesh_to_msg,
)
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.spatial import Pose3D

if TYPE_CHECKING:
    from robotics_utils.motion_planning import MotionPlanningQuery
    from robotics_utils.states import ObjectCentricState, ObjectKinematicState


class PlanningSceneManager:
    """A manager to update the state of the MoveIt planning scene."""

    def __init__(self, *, planning_frame: str) -> None:
        """Initialize an interface for the MoveIt planning scene.

        :param planning_frame: Reference frame used by MoveIt for motion planning
        """
        self.planning_frame = planning_frame
        self.planning_scene = PlanningSceneInterface()
        rospy.sleep(3)  # Allow time for the scene to initialize

        self._added_objects: set[str] = set()
        """Names of all objects added to the planning scene (doesn't include hidden objects)."""

        self._hidden_objects: dict[str, CollisionObjectMsg] = {}
        """Hidden objects are ignored for the purposes of collision checking."""

        self._attached_objects: dict[tuple[str, str], set[str]] = defaultdict(set)
        """Map (robot name, end-effector link name) pairs to the names of attached objects."""

    def set_object_state(self, obj_state: ObjectKinematicState) -> bool:
        """Set the kinematic state of an object in the MoveIt planning scene.

        :param obj_state: Kinematic state of the object to be updated
        :return: True if the object state was successfully set, else False
        """
        msg = self.make_collision_object_msg(obj_state)
        return self.add_object_msg(msg)

    def add_object_msg(self, collision_obj_msg: CollisionObjectMsg) -> bool:
        """Add a moveit_msgs/CollisionObject message to the MoveIt planning scene.

        :param collision_obj_msg: ROS message representing an object's collision geometry
        :return: True if the object was successfully added, else False
        """
        self.planning_scene.add_object(collision_obj_msg)

        object_exists = self.wait_until_object_exists(collision_obj_msg.id)

        if object_exists:
            self._added_objects.add(collision_obj_msg.id)
        else:
            rospy.logerr(f"Failed to add '{collision_obj_msg.id}' to the MoveIt planning scene.")

        return object_exists

    def remove_object(self, obj_name: str) -> bool:
        """Remove the named object from the MoveIt planning scene."""
        if obj_name not in self._added_objects:
            return False

        self.planning_scene.remove_world_object(obj_name)
        removed = self.wait_until_object_removed(obj_name)

        if removed:
            self._added_objects.discard(obj_name)

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
        return object_hidden and (obj_name in self._hidden_objects)

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

    def revert_query_ignores(self, query: MotionPlanningQuery) -> bool:
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

    def attach_object(
        self,
        obj_name: str,
        robot_name: str,
        ee_link_name: str,
        touch_links: list[str],
    ) -> bool:
        """Attach the named object to the given robot's specified end-effector.

        :param obj_name: Name of the object to be attached
        :param robot_name: Name of a robot
        :param ee_link_name: Name of the robot's relevant end-effector link
        :param touch_links: Names of the robot links allowed to touch the attached object
        :return: True if the object was successfully attached, else False
        """
        self.planning_scene.attach_object(obj_name, link=ee_link_name, touch_links=touch_links)
        is_attached = self.wait_until_object_attached(obj_name)

        if is_attached:
            self._attached_objects[(robot_name, ee_link_name)].add(obj_name)
        return is_attached

    def detach_object(self, *, obj_name: str, robot_name: str, ee_link_name: str) -> bool:
        """Detach an object from the specified robot end-effector.

        :param obj_name: Name of the object to be detached
        :param robot_name: Name of the robot that the object is attached to
        :param ee_link_name: Name of the robot's relevant end-effector link
        :return: True if the object was successfully detached, else False
        """
        if obj_name not in self._attached_objects[(robot_name, ee_link_name)]:
            rospy.logwarn(f"Cannot detach unattached object '{obj_name}' from '{robot_name}'.")
            return False

        self.planning_scene.remove_attached_object(link=ee_link_name, name=obj_name)
        is_detached = self.wait_until_object_detached(obj_name)

        if is_detached:
            self._attached_objects[(robot_name, ee_link_name)].remove(obj_name)
        return is_detached

    def get_attached_objects(self, robot_name: str) -> set[str]:
        """Retrieve the names of objects attached to the named robot (defaults to empty set)."""
        all_attached = set()
        for (robot_attached_to, _), objects in self._attached_objects.items():
            if robot_attached_to == robot_name:
                all_attached.update(objects)

        return all_attached

    def get_object_msg(self, obj_name: str) -> CollisionObjectMsg:
        """Retrieve the CollisionObject message for the named object in the planning scene."""
        object_msgs: dict[str, CollisionObjectMsg] = self.planning_scene.get_objects([obj_name])
        return object_msgs[obj_name]

    def get_object_pose(self, obj_name: str) -> Pose3D:
        """Retrieve the pose of the named object from the MoveIt planning scene."""
        obj_pose_msg = self.planning_scene.get_object_poses([obj_name])[obj_name]
        obj_pose = pose_from_msg(obj_pose_msg)
        return replace(obj_pose, ref_frame=self.planning_frame)

    def set_object_pose(self, obj_name: str, pose: Pose3D) -> bool:
        """Update the pose of the named object in the MoveIt planning scene.

        :return: True if the object pose was successfully set, else False
        """
        pose_p_o = TransformManager.convert_to_frame(pose, self.planning_frame)

        move_object_msg = CollisionObjectMsg()
        move_object_msg.id = obj_name
        move_object_msg.operation = CollisionObjectMsg.MOVE
        move_object_msg.pose = pose_to_msg(pose_p_o)
        move_object_msg.header.frame_id = self.planning_frame
        move_object_msg.header.stamp = rospy.Time.now()

        return self.add_object_msg(move_object_msg)

    def set_state(self, state: ObjectCentricState, attempts_per_obj: int = 3) -> bool:
        """Update the MoveIt planning scene to reflect the given environment state."""
        kinematic_states = state.known_kinematic_states
        known_objects = set(kinematic_states.keys())
        unknown_objects = set(state.object_names).difference(known_objects)

        # Remove any objects in the planning scene that don't have a specified state
        to_be_removed = self._added_objects.difference(known_objects)
        to_be_removed.update(unknown_objects)

        all_removed = True
        for obj_name in to_be_removed:
            if obj_name in self._added_objects:
                removed = self.remove_object(obj_name)
                all_removed = all_removed and removed

        # Add updated versions of all objects with known kinematic states
        all_added = True
        for kin_state in kinematic_states.values():
            attempts_left = attempts_per_obj
            object_added = False
            while attempts_left and not object_added:
                object_added = self.set_object_state(kin_state)
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

    def make_collision_object_msg(
        self,
        object_state: ObjectKinematicState,
        object_type: str | None = None,
    ) -> CollisionObjectMsg:
        """Construct a moveit_msgs/CollisionObject message using the given data.

        :param object_state: Kinematic state of an object (i.e., its pose and collision model)
        :param object_type: Type of the object (e.g., `"box"`)
        :return: Constructed moveit_msgs/CollisionObject message
        """
        # Convert object pose into the target frame
        pose_t_o = TransformManager.convert_to_frame(object_state.pose, self.planning_frame)

        msg = CollisionObjectMsg()
        msg.id = object_state.name
        msg.header.frame_id = self.planning_frame
        msg.header.stamp = rospy.Time.now()
        msg.operation = CollisionObjectMsg.ADD

        if object_type is not None:
            msg.type.key = object_type  # Ignore 'db' field of message

        obj_pose_msg = pose_to_msg(pose_t_o)

        msg.meshes = [trimesh_to_msg(mesh) for mesh in object_state.collision_model.meshes]
        msg.mesh_poses = [obj_pose_msg for _ in msg.meshes]

        msg.primitives = [
            primitive_shape_to_msg(ps) for ps in object_state.collision_model.primitives
        ]
        shape_local_poses = []
        for ps in object_state.collision_model.primitives:
            z_size_m = ps.aabb.max_xyz.z - ps.aabb.min_xyz.z
            local_pose = Pose3D.from_xyz_rpy(z=z_size_m / 2.0, ref_frame=object_state.name)
            shape_local_poses.append(local_pose)

        # Compose each primitive shape's local pose with the object's pose in the target frame
        msg.primitive_poses = [pose_to_msg(pose_t_o @ pose_o_s) for pose_o_s in shape_local_poses]

        # Deliberately DO NOT set msg.pose when adding an object

        return msg
