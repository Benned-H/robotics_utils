"""Unit tests for the KinematicTree class."""

from pathlib import Path

import pytest

from robotics_utils.collision_models import Cylinder
from robotics_utils.kinematics import Pose3D
from robotics_utils.kinematics.kinematic_tree import KinematicTree


@pytest.fixture
def environment_yaml() -> Path:
    """Specify a path to an example environment YAML file."""
    env_yaml = Path(__file__).parent / "test_data/yaml/example_environment.yaml"
    assert env_yaml.exists(), f"Expected to find file: {env_yaml}"
    return env_yaml


def test_kinematic_tree_from_yaml(environment_yaml: Path) -> None:
    """Verify that a KinematicTree can be loaded from an example YAML file."""
    tree = KinematicTree.from_yaml(environment_yaml)

    # Assert - Expect that the tree matches what's specified in the YAML file
    assert tree.root_frame == "custom_default"
    assert len(tree.robot_base_poses) == 1
    assert len(tree.object_poses) == 2
    assert len(tree.waypoints) == 2

    spot_base_pose = tree.get_robot_base_pose("spot")
    assert spot_base_pose.approx_equal(Pose3D.identity("custom_default"))

    table1_pose = tree.get_object_pose("table1")
    assert table1_pose.approx_equal(Pose3D.from_xyz_rpy(x=3, ref_frame="custom_default"))

    bottle1_pose = tree.get_object_pose("bottle1")
    assert bottle1_pose.approx_equal(Pose3D.from_xyz_rpy(z=1, ref_frame="table1"))

    origin_pose = tree.waypoints.get("origin").to_3d()
    assert origin_pose.approx_equal(Pose3D.identity("custom_default"))

    past_table_pose = tree.waypoints.get("past_table").to_3d()
    assert past_table_pose.approx_equal(
        Pose3D.from_xyz_rpy(x=3.5, yaw_rad=3.14, ref_frame="custom_default"),
    )

    table_model = tree.get_collision_model("table1")
    assert len(table_model.meshes) == 1
    assert not table_model.primitives

    bottle_model = tree.get_collision_model("bottle1")
    assert len(bottle_model.primitives) == 2
    assert not bottle_model.meshes
    for primitive in bottle_model.primitives:
        assert isinstance(primitive, Cylinder)

    # Because the root frame has no attached geometry, its collision model should be None
    assert tree.get_collision_model(tree.root_frame) is None
