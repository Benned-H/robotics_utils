"""Define strategies for generating kinematics data for property-based testing."""

import hypothesis.strategies as st

from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.poses import Pose2D, Pose3D
from robotics_utils.kinematics.rotations import EulerRPY, Quaternion


@st.composite
def angles_rad(draw: st.DrawFn) -> float:
    """Generate random angles (in radians)."""
    return draw(st.floats(min_value=-10e4, max_value=10e4, allow_infinity=False, allow_nan=False))


@st.composite
def positions(draw: st.DrawFn) -> Point3D:
    """Generate random (x,y,z) points."""
    x = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    y = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    z = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    return Point3D(x, y, z)


@st.composite
def quaternions(draw: st.DrawFn) -> Quaternion:
    """Generate random unit quaternions."""
    x = draw(st.floats(min_value=-10e6, max_value=10e6, allow_infinity=False, allow_nan=False))
    y = draw(st.floats(min_value=-10e6, max_value=10e6, allow_infinity=False, allow_nan=False))
    z = draw(st.floats(min_value=-10e6, max_value=10e6, allow_infinity=False, allow_nan=False))
    return Quaternion(x, y, z, w=1.0)


@st.composite
def poses_3d(draw: st.DrawFn) -> Pose3D:
    """Generate random relative poses in 3D space."""
    position = draw(positions())
    orientation = draw(quaternions())
    ref_frame = draw(st.text())
    return Pose3D(position, orientation, ref_frame)


@st.composite
def poses_2d(draw: st.DrawFn) -> Pose2D:
    """Generate random relative poses on the 2D plane."""
    x = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    y = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    yaw_rad = draw(angles_rad())
    ref_frame = draw(st.text())
    return Pose2D(x, y, yaw_rad, ref_frame)


@st.composite
def euler_rpys(draw: st.DrawFn) -> EulerRPY:
    """Generate random 3D rotations represented as Euler RPY angles."""
    roll_rad = draw(angles_rad())
    pitch_rad = draw(angles_rad())
    yaw_rad = draw(angles_rad())
    return EulerRPY(roll_rad, pitch_rad, yaw_rad)
