"""Define a class to represent a bounding box in an image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from robotics_utils.vision.pixel_xy import PixelXY

if TYPE_CHECKING:
    from robotics_utils.vision.rgb_image import RGBImage
    from robotics_utils.vision.vision_utils import RGB


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular bounding box in an image."""

    top_left: PixelXY
    """Top-left pixel inside the bounding box."""

    bottom_right: PixelXY
    """Bottom-right pixel inside the bounding box."""

    @classmethod
    def from_center(cls, center_pixel: PixelXY, *, width: int, height: int) -> BoundingBox:
        """Construct a bounding box from a center (x,y) pixel, a width, and a height.

        :param center_pixel: Center pixel of the bounding box as an (x,y) image coordinate
        :param width: Width of the bounding box (in pixels)
        :param height: Height of the bounding box (in pixels)
        :return: Constructed BoundingBox instance
        """
        pixels_up = np.floor((height - 1) / 2)  # Even height => Center biases high
        pixels_down = np.ceil((height - 1) / 2)
        pixels_left = np.floor((width - 1) / 2)  # Even width => Center biases left
        pixels_right = np.ceil((width - 1) / 2)

        top_left = center_pixel - PixelXY((pixels_left, pixels_up))
        bottom_right = center_pixel + PixelXY((pixels_right, pixels_down))

        return BoundingBox(top_left, bottom_right)

    @property
    def width(self) -> int:
        """Compute the width (in pixels) of the bounding box."""
        return int(self.bottom_right.x - self.top_left.x) + 1

    @property
    def height(self) -> int:
        """Compute the height (in pixels) of the bounding box."""
        return int(self.bottom_right.y - self.top_left.y) + 1

    @property
    def area_square_px(self) -> int:
        """Compute the area of the bounding box (in square pixels)."""
        return self.width * self.height

    @property
    def center_pixel(self) -> PixelXY:
        """Compute the center pixel of the bounding box as an (x,y) coordinate."""
        return PixelXY(np.floor((self.top_left.xy + self.bottom_right.xy) / 2))

    def draw(self, image: RGBImage, color: RGB, thickness: int = 3) -> None:
        """Draw the bounding box as a rectangle on the given image.

        :param image: Image on which the bounding box is drawn (modified in-place)
        :param color: RGB color of the drawn bounding box
        :param thickness: Thickness (pixels) of the drawn bounding box
        """
        cv2.rectangle(
            image.data,
            tuple(self.top_left.xy),
            tuple(self.bottom_right.xy),
            color,
            thickness,
        )
        cv2.circle(image.data, tuple(self.center_pixel.xy), 1, color, thickness)
