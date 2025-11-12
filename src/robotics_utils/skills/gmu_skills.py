"""Define dataclasses to structure MoveIt-based skills."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

import rospy
from spot_skills.srv import GetRGBDPairs, GetRGBDPairsRequest, GetRGBDPairsResponse, NameService
from std_srvs.srv import Trigger

from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D
from robotics_utils.kinematics.rotations import Quaternion

from robotics_utils.ros.manipulator import Manipulator
from robotics_utils.ros.services import ServiceCaller, trigger_service
from robotics_utils.ros.transform_manager import TransformManager


@dataclass(frozen=True)
class LookParameters:
    """Parameters defining a 'Look' trajectory for an end-effector camera."""

    target_pose_ee: Pose3D
    camera_name: str


@dataclass(frozen=True)
class PickParameters:
    """Parameters defining a 'Pick' trajectory w.r.t. some object."""

    pose_o_g: Pose3D
    """Grasp pose (g) of the end-effector w.r.t. the object (o)."""
    x_pre_grasp_offset_m: float = 0
    y_pre_grasp_offset_m: float = 0
    z_pre_grasp_offset_m: float = 0

    x_post_grasp_lift_m: float = 0
    y_post_grasp_lift_m: float = 0
    z_post_grasp_lift_m: float = 0
    body_frame_name: str = "body"
    carry_pose: str = "vert_carry"       #Refers to which carry pose is used from ee_poses.yaml


# def execute_pick(manipulator: Manipulator, params: PickParameters) -> None:
#     """Execute a Pick skill based on the given parameters."""
#     pose_b_g = TransformManager.convert_to_frame(params.pose_o_g, params.body_frame_name)

#     # Open the gripper
#     # manipulator.open_gripper()

#     # Pre-grasp trajectory
#     pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-params.pre_grasp_offset_m)  # Pregrasp w.r.t. grasp
#     pose_b_pregrasp = pose_b_g @ pose_g_pregrasp

#     manipulator.execute_motion_plan(pose_b_pregrasp, params.body_frame_name)

#     # Grasp trajectory
#     manipulator.execute_motion_plan(pose_b_g, params.body_frame_name)

#     # Close the gripper
#     manipulator.close_gripper()

#     # Post-grasp trajectory
#     pose_w_g = TransformManager.convert_to_frame(pose_b_g, DEFAULT_FRAME)
#     pose_w_postgrasp = pose_w_g @ Pose3D.from_xyz_rpy(z=params.post_grasp_lift_m)

#     manipulator.execute_motion_plan(pose_w_postgrasp, params.body_frame_name)


SPOT_PICK_CUP = PickParameters(
    pose_o_g=Pose3D.from_xyz_rpy(x=-0.17, y=0.03, z=0.07, roll_rad=1.5708, ref_frame="waterbottle"),
    x_pre_grasp_offset_m=0.15,
    y_post_grasp_lift_m=-0.1,
)

SPOT_PICK_ERASER = PickParameters(
    pose_o_g=Pose3D.from_xyz_rpy(x=-0.00, y=0.00, z=0.24, pitch_rad=1.5708, ref_frame="cellphone"),
    x_pre_grasp_offset_m=0.1,
    x_post_grasp_lift_m=0.1,
    carry_pose="horiz_carry"
)

def execute_look(manipulator: Manipulator, lookpose) -> None:
    manipulator.execute_motion_plan(lookpose, "body")
    manipulator.open_gripper()

    # Take photo!
    time.sleep(4)

    return



def execute_pick(manipulator: Manipulator, params: PickParameters) -> None:
    """Execute a Pick skill based on the given parameters."""
    pose_b_g = TransformManager.convert_to_frame(params.pose_o_g, params.body_frame_name)

    # Open the gripper
    # manipulator.open_gripper()

    # Pre-grasp trajectory
    pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-params.x_pre_grasp_offset_m, 
                                          y=-params.y_pre_grasp_offset_m,
                                          z=-params.z_pre_grasp_offset_m)  # Pregrasp w.r.t. grasp
    pose_b_pregrasp = pose_b_g @ pose_g_pregrasp

    manipulator.execute_motion_plan(pose_b_pregrasp, params.body_frame_name)

    # Grasp trajectory
    manipulator.execute_motion_plan(pose_b_g, params.body_frame_name)

    # Close the gripper
    manipulator.close_gripper()

    # Post-grasp trajectory
    pose_w_g = TransformManager.convert_to_frame(pose_b_g, DEFAULT_FRAME)
    pose_w_postgrasp = pose_w_g @ Pose3D.from_xyz_rpy(x=-params.x_post_grasp_lift_m,
                                                      y=-params.y_post_grasp_lift_m,
                                                      z=-params.z_post_grasp_lift_m)

    manipulator.execute_motion_plan(pose_w_postgrasp, params.body_frame_name)


Pickable = NewType("Pickable", str)

waterbottle_place_params = {"countertop": {"preplace": ["cup_preplace_countertop1", "cup_preplace_countertop2"],
                                     "postplace": ["cup_postplace_countertop1",]},
                                     }

cellphone_place_params = {"desk": {"preplace": ["eraser_preplace_desk1", "eraser_preplace_desk2"],
                                   "postplace": ["eraser_postplace_desk1",]},
                          "bed":  {"preplace": ["eraser_preplace_bed1", "eraser_preplace_bed2"],
                                   "postplace": ["eraser_postplace_bed1",]},
                        }



class SpotSkillsExecutor:
    """Interface for executing skills on Spot."""

    def __init__(self) -> None:
        """Define skill parameters for each object."""
        self.pick_params: dict[str, PickParameters] = {"waterbottle": SPOT_PICK_CUP, 
                                                       "cellphone": SPOT_PICK_ERASER}
        
        self.pre_place_params: dict[str, [str]] = {"waterbottle": ["cup_pre_place", "cup_place_countertop"], 
                                                "cellphone": ["eraser_pre_place", "eraser_place_desk"]}
        
        self.post_place_params: dict[str, [str]] = {"waterbottle": ["cup_post_place_1",], 
                                                "cellphone": ["eraser_post_place_1"]}
        
        self.place_params = {"waterbottle": waterbottle_place_params,
                             "cellphone": cellphone_place_params,
                             }
        

        self.spot_arm = Manipulator(move_group="arm", base_link="body", grasping_group="gripper")
        self.looking = True
        self.robot_location = "robot_start_pose"

        self.ee_poses = Pose3D.load_named_poses(
            Path("/resources/apriltags/ee_poses.yaml"),
            "ee_poses",
        )

        self.look_container_dict = {"kitchen" : "high",
                                    "garbagecan" : "trash",
                                    "diningtable" : "high",
                                    "fridge" : "higher",
                                    "countertop" : "high",
                                    "sink" : "medium",
                                    "bedroom" : "high",
                                    "couch" : "high",
                                    "desk" : "wide",
                                    "tvstand" : "high",
                                    "bed" : "high",
                                    "robot_start_pose": "high", #This is an explicitly bad solution to the planner calling look and not knowing where the robot is
                                    }

        trigger_service("spot/unlock_arm")
        # trigger_service("spot/stow_arm")
        self._command_dictionary = {
            "0": [self.retract],
            "l": [self.look],
            "1": [self.look, "trash", ["waterbottle", "cellphone"]],
            "2": [self.look, "high",["waterbottle", "cellphone"]],
            "2a": [self.look, "higher",["waterbottle", "cellphone"]],
            "3": [self.look, "medium",["waterbottle", "cellphone"]],
            "3a": [self.look, "wide",["waterbottle", "cellphone"]],
            "d": [self.move_spot_move_base, "dock"],
            "eb": [self.move_spot_move_base, "enter_bedroom"],

            "4": [self.move_spot_move_base, "kitchen"],
            "4r": [self.move_spot_move_base, "reverse_kitchen"],
            "5": [self.move_spot_move_base, "garbagecan"],
            "6": [self.move_spot_move_base, "diningtable"],
            "7": [self.move_spot_move_base, "fridge"],
            "8": [self.move_spot_move_base, "countertop"],
            "9": [self.move_spot_move_base, "sink"],
            "10": [self.move_spot_move_base, "bedroom"],
            "10r": [self.move_spot_move_base, "reverse_bedroom"],
            "11": [self.move_spot_move_base, "couch"],
            "12": [self.move_spot_move_base, "desk"],
            "13": [self.move_spot_move_base, "tvstand"],
            "14": [self.move_spot_move_base, "bed"],
            "15": [self.pick, "waterbottle"],
            "16": [self.place, "waterbottle", "countertop"],
            "17": [self.pick, "cellphone"],
            "18": [self.place, "cellphone", "desk"],
            "19": [self.place, "cellphone", "bed"],

        }

    def place(self, object, place):
        print(self.place_params[object][place])
        for ee in self.place_params[object][place]["preplace"]:
            self.motion_plan_to_pose(ee)

        self.spot_arm.open_gripper()
        for ee in self.place_params[object][place]["postplace"]:
            self.motion_plan_to_pose(ee)
        self.retract()

    def pick(self, picked: Pickable) -> None:
        """Pick an object with the given name."""
        
        if not self.looking:
            height = self.look_container_dict[self.robot_location]
            look_ee_pose = self.ee_poses.get(height) #picked objects (in this demo) are always seen from this pose
            pose_b_look = TransformManager.convert_to_frame(look_ee_pose, "body")
            execute_look(manipulator=self.spot_arm, lookpose=pose_b_look)

        self.looking = False
        execute_pick(self.spot_arm, self.pick_params[picked])

        carry_ee_pose = self.ee_poses.get(self.pick_params[picked].carry_pose)
        if carry_ee_pose is None:
            rospy.logwarn("No 'carry' pose.")
            exit()

        self.spot_arm.execute_motion_plan(carry_ee_pose, "body")

    def look(self, height: str = "auto", missing_objects: list = []) -> list[str]:
        """Look into a container of the given height."""
        if height == "auto":
            print("----------looking at:---------")
            print(self.robot_location)
            height = self.look_container_dict[self.robot_location]
        look_ee_pose = self.ee_poses.get(height)
        if look_ee_pose is None:
            rospy.logwarn(f"Height '{height}' had no corresponding 'Look' pose.")
            exit()

        pose_b_look = TransformManager.convert_to_frame(look_ee_pose, "body")
        execute_look(manipulator=self.spot_arm, lookpose=pose_b_look)
        self.looking=True

        found_objects = []
        for obj in missing_objects:
            pose = TransformManager.lookup_transform(source_frame=obj, target_frame="body", timeout_s=0.3)
            if pose is not None:
                found_objects.append(obj)
        # if len(found_objects) == 0:
        # self.retract()
        return found_objects

        # rgbd_getter = ServiceCaller[GetRGBDPairsRequest, GetRGBDPairsResponse](
        #     "spot/get_rgbd_pairs",
        #     GetRGBDPairs,
        # )

    def retract(self) -> None:
        trigger_service("spot/stow_arm")
        self.spot_arm.close_gripper()

        self.looking = False


    def oldplace(self, object: Pickable) -> None:
        """Place an object."""
        # for ee in ["cup_pre_place", "cup_place_countertop"]:
        for ee in self.pre_place_params[object]:
            self.motion_plan_to_pose(ee)

        self.spot_arm.open_gripper()
        for ee in self.post_place_params[object]:
            self.motion_plan_to_pose(ee)
        self.retract()

    def motion_plan_to_pose(self, ee_pose_name: str) -> None:
        """Motion plan to a pre-defined end effector pose."""
        target_ee_pose = self.ee_poses.get(ee_pose_name)
        if target_ee_pose is None:
            rospy.logwarn(f"No '{ee_pose_name}' pose.")
            exit()

        self.spot_arm.execute_motion_plan(target_ee_pose, "body")

    def move_spot_move_base(self, container_name: str) -> bool:
        """_summary_  moves the robot body to a predefined pose relative to a named container.

        :param container_name: _description_
        :return: _description_
        """
        if self.looking:
            self.retract()
            time.sleep(1)


        rospy.wait_for_service("/spot/navigation/to_waypoint")
        self.robot_location = container_name

        try:
            move_to_waypoint = rospy.ServiceProxy("/spot/navigation/to_waypoint", NameService)
            response = move_to_waypoint(name=container_name)
            self.robot_location = container_name
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call /spot/navigation/to_waypoint: %s", e)
            return False

    def test_skills(self) -> None:
        while True:
            print("""
            Options:
            (0)retract arm
            (l)auto-look
            (1)look trash
            (2)look high
            (3)look medium
            (d)move dock
            (4)move kitchen
            (5)move garbagecan
            (6)move diningtable
            (7)move fridge
            (8)move countertop
            (9)move sink
            (10)move bedroom
            (11)move couch
            (12)move desk
            (13)move tvstand
            (14)move bed
            (15)pick cup 
            (16)place cup on countertop
            (17)pick eraser 
            (18)place eraser on desk
            (19)place eraser on bed

            (q) Exit.
            """)

            try:  # noqa: SIM105
                inputs = input(">")
            except NameError:
                pass
            print(inputs)
            req_type = str.split(inputs)[0]

            if req_type == "q":
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                skill = cmd_func[0]
                skill_params = cmd_func[1:]
                print(skill)
                print(skill_params)
                skill(*skill_params)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    # exit_code = 0
    # if not main(sys.argv[1:]):
    #     exit_code = 1
    # os._exit(exit_code)

    from moveit_commander import roscpp_initialize

    TransformManager.init_node("taskplan")

    roscpp_initialize(sys.argv)

    executor = SpotSkillsExecutor()
    executor.test_skills()
