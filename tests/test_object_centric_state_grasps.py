"""Unit tests for ObjectCentricState grasp accessors and known-pose helpers."""

from robotics_utils.spatial import Pose3D
from robotics_utils.states import GraspAttachment, ObjectCentricState


def test_grasp_attach_query_detach_lifecycle() -> None:
    """Verify grasp attachments can be attached, queried, and detached by object name."""
    state = ObjectCentricState(robot_names={"spot"}, object_names={"obj1"}, root_frame="map")
    state.add_end_effector(robot_name="spot", ee_link_name="gripper_link")

    grasp = GraspAttachment(
        obj_name="obj1",
        robot_name="spot",
        ee_link_name="gripper_link",
        pose_ee_o=Pose3D.from_xyz_rpy(x=0.02, y=0.01, z=0.25, ref_frame="gripper_link"),
        touching_link_names={"gripper_link", "finger_link"},
    )

    state.attach_grasp(grasp)

    assert state.get_grasp_for_object("obj1") == grasp
    assert state.grasp_attachments == (grasp,)

    detached = state.detach_grasp_for_object("obj1")
    assert detached == grasp
    assert state.get_grasp_for_object("obj1") is None
    assert state.grasp_attachments == ()


def test_known_object_poses_only_include_known_sources() -> None:
    """Verify known-object pose accessor excludes estimated-only poses."""
    state = ObjectCentricState(
        robot_names={"spot"},
        object_names={"known_obj", "estimated_obj"},
        root_frame="map",
    )

    state.set_known_object_pose("known_obj", Pose3D.from_xyz_rpy(x=1.0, ref_frame="map"))
    state.set_estimated_object_pose("estimated_obj", Pose3D.from_xyz_rpy(x=2.0, ref_frame="map"))

    known_poses = state.known_object_poses
    assert set(known_poses.keys()) == {"known_obj"}


def test_has_end_effector_reflects_registered_links() -> None:
    """Verify end-effector registration and lookup helper behavior."""
    state = ObjectCentricState(robot_names={"spot"}, object_names={"obj1"}, root_frame="map")

    assert not state.has_end_effector(robot_name="spot", ee_link_name="gripper_link")

    state.add_end_effector(robot_name="spot", ee_link_name="gripper_link")

    assert state.has_end_effector(robot_name="spot", ee_link_name="gripper_link")
