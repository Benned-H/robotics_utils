"""Define strategies for generating vision data for property-based testing."""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as numpy_st

from robotics_utils.vision.bounding_box import BoundingBox
from robotics_utils.vision.images import DepthImage, Image, PixelXY, RGBImage

from ..common_strategies import integer_ranges


@st.composite
def pixels_xy(draw: st.DrawFn) -> PixelXY:
    """Generate random (x,y) pixel coordinates."""
    x = draw(st.integers(min_value=-100000, max_value=100000))
    y = draw(st.integers(min_value=-100000, max_value=100000))

    return PixelXY((x, y))


@st.composite
def bounding_boxes(draw: st.DrawFn) -> BoundingBox:
    """Generate random bounding boxes in pixel coordinates."""
    min_x, max_x = draw(integer_ranges(min_value=-100000, max_value=100000))
    min_y, max_y = draw(integer_ranges(min_value=-100000, max_value=100000))

    top_left = PixelXY((min_x, min_y))
    bottom_right = PixelXY((max_x, max_y))

    return BoundingBox(top_left, bottom_right)


@st.composite
def rgb_images(draw: st.DrawFn) -> RGBImage:
    """Generate random RGB images."""
    height = draw(st.integers(min_value=1, max_value=1024))
    width = draw(st.integers(min_value=1, max_value=1024))

    data = draw(numpy_st.arrays(dtype=np.uint8, shape=(height, width, 3)))
    return RGBImage(data)


@st.composite
def depth_images(draw: st.DrawFn) -> DepthImage:
    """Generate random depth images."""
    height = draw(st.integers(min_value=1, max_value=1024))
    width = draw(st.integers(min_value=1, max_value=1024))

    data = draw(numpy_st.arrays(dtype=np.float64, shape=(height, width)))
    return DepthImage(data)


@st.composite
def images(draw: st.DrawFn) -> Image:
    """Generate random RGB or depth images."""
    return draw(st.one_of(rgb_images(), depth_images()))
