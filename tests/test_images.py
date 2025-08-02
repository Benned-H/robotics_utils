"""Unit tests for image-related classes defined in images.py."""

from __future__ import annotations

from hypothesis import given

from robotics_utils.vision.images import PixelXY, RGBImage

from .vision_strategies import pixels_xy, rgb_images


@given(rgb_images(), pixels_xy())
def test_image_clip_pixel(image: RGBImage, pixel: PixelXY) -> None:
    """Verify that any image correctly clips (x,y) pixels into its image indices."""
    # Arrange/Act - Given an image and (x,y) coordinates, clip the (x,y) pixel into the image
    result_pixel = image.clip_pixel(pixel)

    # Assert - Expect that the shifted pixel's coordinates are inside the image
    assert 0 <= result_pixel.x <= image.width - 1
    assert 0 <= result_pixel.y <= image.height - 1
