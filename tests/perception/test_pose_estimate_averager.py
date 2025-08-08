"""Unit tests for the PoseEstimateAverager class."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

from robotics_utils.kinematics import Pose3D
from robotics_utils.perception.pose_estimation import PoseEstimateAverager

from ..kinematics.kinematics_strategies import poses_3d


@st.composite
def pose_estimates_map(draw: st.DrawFn) -> dict[str, list[Pose3D]]:
    """Generate a map of random frame names to random poses."""
    return draw(
        st.dictionaries(
            keys=st.text(),
            values=st.lists(elements=poses_3d(), max_size=500),
            max_size=50,
        ),
    )


@given(pose_estimates_map(), st.integers(min_value=1, max_value=1000))
def test_pose_estimate_averager_compute_all_averages(
    poses_map: dict[str, list[Pose3D]],
    window_size: int,
) -> None:
    """Verify that all frames have averaged pose estimates after all averages are computed."""
    # Arrange - Populate a pose estimate averager using the given map of pose estimates
    averager = PoseEstimateAverager(window_size=window_size)
    for frame_name, estimates in poses_map.items():
        for pose in estimates:
            averager.update(frame_name, pose)

    # Act - Compute all averages using the averager
    results = averager.compute_all_averages()

    # Assert - Expect that all frames with pose estimates have an average
    for frame_name, estimates in poses_map.items():
        if estimates:
            assert results[frame_name] is not None
        else:
            assert frame_name not in results
