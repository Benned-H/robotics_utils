"""Define strategies for generating geometric primitives for property-based testing."""

import hypothesis.strategies as st

from robotics_utils.geometry import AxisAlignedBoundingBox, Point3D

from .common_strategies import real_ranges


@st.composite
def positions(draw: st.DrawFn) -> Point3D:
    """Generate random (x,y,z) points."""
    x = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    y = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    z = draw(st.floats(min_value=-10e8, max_value=10e8, allow_infinity=False, allow_nan=False))
    return Point3D(x, y, z)


@st.composite
def aabbs(draw: st.DrawFn) -> AxisAlignedBoundingBox:
    """Generate random axis-aligned bounding boxes."""
    min_x, max_x = draw(real_ranges(finite=False))
    min_y, max_y = draw(real_ranges(finite=False))
    min_z, max_z = draw(real_ranges(finite=False))

    min_xyz = Point3D(min_x, min_y, min_z)
    max_xyz = Point3D(max_x, max_y, max_z)

    return AxisAlignedBoundingBox(min_xyz, max_xyz)
