"""Unit tests for image-related classes defined in images.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hypothesis import given

from .strategies.vision_strategies import images, pixels_xy

if TYPE_CHECKING:
    from robotics_utils.vision import Image, PixelXY


@given(images(), pixels_xy())
def test_image_clip_pixel(image: Image, pixel: PixelXY) -> None:
    """Verify that any image correctly clips (x,y) pixels into its image indices."""
    # Arrange/Act - Given an image and (x,y) coordinates, clip the (x,y) pixel into the image
    result_pixel = image.clip_pixel(pixel)

    # Assert - Expect that the shifted pixel's coordinates are inside the image
    assert 0 <= result_pixel.x <= image.width - 1
    assert 0 <= result_pixel.y <= image.height - 1
