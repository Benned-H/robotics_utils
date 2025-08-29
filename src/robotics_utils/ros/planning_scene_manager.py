"""Define a class to synchronize the MoveIt planning scene with an external kinematic state."""

import time

import rospy
from moveit_commander import PlanningSceneInterface

from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.ros.msg_conversion import make_collision_object_msg
from robotics_utils.ros.transform_manager import TransformManager


class PlanningSceneManager:
    """A manager to update the state of the MoveIt planning scene."""

    def __init__(self, tree: KinematicTree) -> None:
        """Initialize the scene manager using the given environment state."""
        self.planning_scene = PlanningSceneInterface()
        rospy.sleep(3)  # Allow time for the scene to initialize
        self.set_planning_scene(tree)

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

            collision_msg = make_collision_object_msg(object_name, "", pose_b_o, collision_model)
            self.planning_scene.add_object(collision_msg)

            if not self.wait_until_object_exists(object_name):
                rospy.logerr(f"Failed to add object '{object_name}' to the MoveIt planning scene.")

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
