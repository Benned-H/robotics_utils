"""Define strategies for generating kinematics data for property-based testing."""

import hypothesis.strategies as st

from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.pose3d import Pose3D
from robotics_utils.kinematics.rotations import Quaternion


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
def poses(draw: st.DrawFn) -> Pose3D:
    """Generate random relative poses in 3D space."""
    position = draw(positions())
    orientation = draw(quaternions())
    ref_frame = draw(st.text())
    return Pose3D(position, orientation, ref_frame)
