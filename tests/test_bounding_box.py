"""Unit tests for the BoundingBox class."""

import hypothesis.strategies as st
from hypothesis import given

from robotics_utils.vision.bounding_box import BoundingBox
from robotics_utils.vision.images import PixelXY, RGBImage

from .vision_strategies import bounding_boxes, pixels_xy, rgb_images


@given(bounding_boxes(), rgb_images())
def test_bounding_box_crop_image(bounding_box: BoundingBox, image: RGBImage) -> None:
    """Verify that any bounding box crop of an image results in the expected dimensions."""
    # Arrange/Act - Given a bounding box and an RGB image, crop the image based on the bounding box
    crop_result = bounding_box.crop(image)

    # Assert - Verify that the resulting image has the expected height and width
    clipped_top_left = image.clip_pixel(bounding_box.top_left)
    clipped_bottom_right = image.clip_pixel(bounding_box.bottom_right)

    assert clipped_top_left.x <= clipped_bottom_right.x
    assert clipped_top_left.y <= clipped_bottom_right.y

    min_x, min_y = clipped_top_left
    max_x, max_y = clipped_bottom_right

    assert crop_result.width == (max_x - min_x + 1)
    assert crop_result.height == (max_y - min_y + 1)


@given(
    pixels_xy(),
    st.integers(min_value=1, max_value=100000),
    st.integers(min_value=1, max_value=100000),
)
def test_bounding_box_from_center(center_pixel: PixelXY, height: int, width: int) -> None:
    """Verify that any bounding box constructed from a center pixel has the expected dimensions."""
    # Arrange/Act - Construct a BoundingBox instance using the given inputs
    result_bb = BoundingBox.from_center(center_pixel, height, width)

    # Assert - Expect that the bounding box has a correct center pixel, height, and width
    diff_x = center_pixel.x - result_bb.center_pixel.x
    diff_y = center_pixel.y - result_bb.center_pixel.y
    if diff_x or diff_y:
        print(f"Center pixel: {center_pixel}")
        print(f"Result center pixel: {result_bb.center_pixel}")
        print(f"Difference in x: {center_pixel.x - result_bb.center_pixel.x}")
        print(f"Difference in y: {center_pixel.y - result_bb.center_pixel.y}")

    assert result_bb.center_pixel == center_pixel
    assert result_bb.height == height
    assert result_bb.width == width
