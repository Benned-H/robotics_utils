"""Define strategies for generating vision data for property-based testing."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis.extra import numpy as numpy_st

from robotics_utils.vision.bounding_box import BoundingBox
from robotics_utils.vision.images import PixelXY, RGBImage


@st.composite
def pixels_xy(draw: st.DrawFn) -> PixelXY:
    """Generate random (x,y) pixel coordinates."""
    x = draw(st.integers(min_value=-100000, max_value=100000))
    y = draw(st.integers(min_value=-100000, max_value=100000))

    return PixelXY((x, y))


@st.composite
def bounding_boxes(draw: st.DrawFn) -> BoundingBox:
    """Generate random bounding boxes in pixel coordinates."""
    x1 = draw(st.integers(min_value=-100000, max_value=100000))
    x2 = draw(st.integers(min_value=-100000, max_value=100000))
    y1 = draw(st.integers(min_value=-100000, max_value=100000))
    y2 = draw(st.integers(min_value=-100000, max_value=100000))

    top_left = PixelXY((min(x1, x2), min(y1, y2)))
    bottom_right = PixelXY((max(x1, x2), max(y1, y2)))

    return BoundingBox(top_left, bottom_right)


@st.composite
def rgb_images(draw: st.DrawFn) -> RGBImage:
    """Generate random RGB images."""
    height = draw(st.integers(min_value=1, max_value=1024))
    width = draw(st.integers(min_value=1, max_value=1024))

    data = draw(numpy_st.arrays(dtype=int, shape=(height, width, 3), elements=st.integers(0, 255)))
    return RGBImage(data)
