"""Define a class to synchronize the MoveIt planning scene with an external kinematic state."""

import time

import rospy
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import CollisionObject

from robotics_utils.collision_models.collision_model import CollisionModel
from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.ros.msg_conversion import make_collision_object_msg, pose_from_msg, pose_to_msg
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.world_models.simulators import ObjectModel, Simulator


class PlanningSceneManager(Simulator):
    """A manager to update the state of the MoveIt planning scene."""

    def __init__(self, tree: KinematicTree) -> None:
        """Initialize the scene manager using the given environment state."""
        self.planning_scene = PlanningSceneInterface()
        rospy.sleep(3)  # Allow time for the scene to initialize
        self.set_planning_scene(tree)

        self._hidden_objects: dict[str, CollisionObject] = {}
        """Hidden objects are ignored for the purposes of collision checking."""

    def add_object(self, obj_model: ObjectModel) -> bool:
        """Add an object to the MoveIt planning scene.

        :param obj_model: Geometric model of the object to be added
        :return: True if the object was successfully added, else False
        """
        TransformManager.broadcast_transform(obj_model.name, obj_model.pose)
        pose_b_o = TransformManager.convert_to_frame(obj_model.pose, "body")  # Object w.r.t. body

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

        return object_exists

    def remove_object(self, obj_name: str) -> bool:
        """Remove the named object from the MoveIt planning scene.

        :param obj_name: Name of the object to be removed
        :return: True if the object was successfully removed, else False
        """
        self.planning_scene.remove_world_object(obj_name)
        return self.wait_until_object_removed(obj_name)

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

    def get_object_msg(self, obj_name: str) -> CollisionObject:
        """Retrieve the CollisionObject message for the named object in the planning scene."""
        object_msgs: dict[str, CollisionObject] = self.planning_scene.get_objects([obj_name])
        return object_msgs[obj_name]

    def set_object_pose(self, obj_name: str, new_pose: Pose3D) -> None:
        """Update the pose of the named object in the MoveIt planning scene."""
        new_pose_b_o = TransformManager.convert_to_frame(new_pose, "body")  # New pose w.r.t. body

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

    def set_planning_scene(self, tree: KinematicTree) -> None:
        """Update the MoveIt planning scene to reflect the given environment state."""
        for object_name in tree.object_names:
            pose_b_o = TransformManager.lookup_transform(object_name, "body")  # Object w.r.t. body
            if pose_b_o is None:
                rospy.logwarn(
                    f"Omitting object '{object_name}' from the MoveIt planning scene "
                    "because its /tf transform is undefined...",
                )
                continue

            collision_model = tree.get_collision_model(object_name)
            if collision_model is None:
                rospy.logwarn(
                    f"Omitting object '{object_name}' from the MoveIt planning scene "
                    "because its collision model is undefined...",
                )
                continue

            obj_model = ObjectModel(object_name, pose_b_o, collision_model)
            self.add_object(obj_model)

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

        :param name: Name of the object to remove from the planning scene
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is removed in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name not in self.planning_scene.get_known_object_names():
                return True
            time.sleep(0.1)

        return False
