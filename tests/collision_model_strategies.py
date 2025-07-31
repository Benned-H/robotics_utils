"""Define strategies for generating collision model data for property-based testing."""

import hypothesis.strategies as st

from robotics_utils.kinematics.collision_models import AxisAlignedBoundingBox
from robotics_utils.kinematics.point3d import Point3D

from .common_strategies import real_ranges


@st.composite
def aabbs(draw: st.DrawFn) -> AxisAlignedBoundingBox:
    """Generate random axis-aligned bounding boxes."""
    min_x, max_x = draw(real_ranges(finite=False))
    min_y, max_y = draw(real_ranges(finite=False))
    min_z, max_z = draw(real_ranges(finite=False))

    min_xyz = Point3D(min_x, min_y, min_z)
    max_xyz = Point3D(max_x, max_y, max_z)

    return AxisAlignedBoundingBox(min_xyz, max_xyz)
