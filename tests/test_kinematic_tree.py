"""Unit tests for the KinematicTree class."""

from pathlib import Path

import numpy as np
import pytest

from robotics_utils.kinematics.collision_models import Cylinder
from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.kinematics.poses import Pose3D


@pytest.fixture
def environment_yaml() -> Path:
    """Specify a path to an example environment YAML file."""
    env_yaml = Path(__file__).parent / "test_data/example_environment.yaml"
    assert env_yaml.exists(), f"Expected to find file: {env_yaml}"
    return env_yaml


def test_kinematic_tree_from_yaml(environment_yaml: Path) -> None:
    """Verify that a KinematicTree can be loaded from an example YAML file."""
    # Arrange/Act - Load a KinematicTree from the YAML filepath provided via test fixture
    loaded_tree = KinematicTree.from_yaml(environment_yaml)

    # Assert - Expect that the tree matches what's specified in the YAML file
    assert loaded_tree.root_frame == "map"
    assert len(loaded_tree.robot_base_poses) == 1
    assert len(loaded_tree.object_poses) == 2
    assert len(loaded_tree.waypoints) == 1

    spot_base_pose = loaded_tree.get_robot_base_pose("spot")
    assert spot_base_pose.approx_equal(Pose3D.identity("map"))

    table1_pose = loaded_tree.get_object_pose("table1")
    assert table1_pose.approx_equal(Pose3D.from_xyz_rpy(x=3, ref_frame="map"))

    bottle1_pose = loaded_tree.get_object_pose("bottle1")
    assert bottle1_pose.approx_equal(Pose3D.from_xyz_rpy(z=1, ref_frame="table1"))

    past_table_waypoint = loaded_tree.waypoints.get("past_table")

    assert past_table_waypoint.to_3d().approx_equal(
        Pose3D.from_xyz_rpy(x=3.5, yaw_rad=3.14159, ref_frame="map"),
    )

    table_model = loaded_tree.collision_models["table1"]
    assert len(table_model.meshes) == 1
    assert not table_model.primitives

    bottle_model = loaded_tree.collision_models["bottle1"]
    assert len(bottle_model.primitives) == 2
    assert not bottle_model.meshes
    for primitive in bottle_model.primitives:
        assert isinstance(primitive, Cylinder)
