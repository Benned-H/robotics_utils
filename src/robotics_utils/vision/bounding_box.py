"""Define a dataclass to represent a bounding box in an image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from robotics_utils.vision.images import Image, PixelXY, RGBImage

if TYPE_CHECKING:
    from robotics_utils.vision.vision_utils import RGB


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular bounding box in an image."""

    top_left: PixelXY  # Top-left pixel inside the bounding box
    bottom_right: PixelXY  # Bottom-right pixel inside the bounding box

    @classmethod
    def from_ratios(cls, ratios: list[float], image_shape: tuple[int, int, int]) -> BoundingBox:
        """Construct a bounding box from image coordinates represented as ratios.

        :param ratios: Bounding box data specified as ratios across the image
        :param image_shape: Shape (rows, cols, channels) of the relevant image
        :return: Constructed BoundingBox instance
        """
        if len(ratios) != 4:
            raise ValueError(f"Cannot construct BoundingBox from a list of length {len(ratios)}.")

        ratios_arr = np.array(ratios)
        top_left_ratios = ratios_arr[:2]
        bottom_right_ratios = ratios_arr[2:]

        height, width, _ = image_shape
        xy_scale = np.array([width, height])

        top_left = top_left_ratios * xy_scale  # Element-wise multiplication
        bottom_right = bottom_right_ratios * xy_scale

        return BoundingBox(PixelXY(top_left), PixelXY(bottom_right))

    @classmethod
    def from_center(cls, center_pixel: PixelXY, height: int, width: int) -> BoundingBox:
        """Construct a bounding box from a center (x,y) pixel, a height, and a width.

        :param center_pixel: Center pixel of the bounding box as an (x,y) image coordinate
        :param height: Height of the bounding box (in pixels)
        :param width: Width of the bounding box (in pixels)
        :return: Constructed BoundingBox instance
        """
        pixels_up = np.floor((height - 1) / 2)  # Even height => Center biases high
        pixels_down = np.ceil((height - 1) / 2)
        pixels_left = np.floor((width - 1) / 2)  # Even width => Center biases left
        pixels_right = np.ceil((width - 1) / 2)

        top_left = center_pixel.xy - np.array([pixels_left, pixels_up])
        bottom_right = center_pixel.xy + np.array([pixels_right, pixels_down])

        return BoundingBox(PixelXY(top_left), PixelXY(bottom_right))

    @property
    def width(self) -> int:
        """Compute the width (in pixels) of the bounding box."""
        return int(self.bottom_right.x - self.top_left.x) + 1

    @property
    def height(self) -> int:
        """Compute the height (in pixels) of the bounding box."""
        return int(self.bottom_right.y - self.top_left.y) + 1

    @property
    def center_pixel(self) -> PixelXY:
        """Compute the center pixel of the bounding box as an (x,y) coordinate."""
        return PixelXY(np.floor((self.bottom_right.xy + self.top_left.xy) / 2))

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

    def crop(self, image: Image, scale_ratio: float = 1.0) -> Image:
        """Return a crop of the given image based on this bounding box.

        :param image: Image from which a cropped section is taken
        :param scale_ratio: Ratio to scale the bounding box size (defaults to 1.0)
        :return: New image containing the cropped section
        """
        scaled_height = int(self.height * scale_ratio)
        scaled_width = int(self.width * scale_ratio)
        scaled_box = BoundingBox.from_center(self.center_pixel, scaled_height, scaled_width)

        return image.get_crop(scaled_box.top_left, scaled_box.bottom_right)
