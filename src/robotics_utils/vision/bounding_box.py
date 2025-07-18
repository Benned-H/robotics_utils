"""Define a dataclass to represent a bounding box in an image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from robotics_utils.vision.vision_utils import RGB


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular bounding box in an image."""

    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int

    @classmethod
    def from_ratios(cls, ratios: list[float], image_shape: tuple[int, int, int]) -> BoundingBox:
        """Construct a bounding box from image coordinates represented as ratios.

        :param ratios: Bounding box data specified as ratios across the image
        :param image_shape: Shape (rows, cols, channels) of the relevant image
        :return: Constructed BoundingBox instance
        """
        if len(ratios) != 4:
            raise ValueError(f"Cannot construct BoundingBox from a list of length {len(ratios)}.")

        height, width, _ = image_shape
        top_left_x = int(ratios[0] * width)
        top_left_y = int(ratios[1] * height)
        bottom_right_x = int(ratios[2] * width)
        bottom_right_y = int(ratios[3] * height)

        return BoundingBox(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    @property
    def width(self) -> int:
        """Compute the width (in pixels) of the bounding box."""
        return self.bottom_right_x - self.top_left_x

    @property
    def height(self) -> int:
        """Compute the height (in pixels) of the bounding box."""
        return self.bottom_right_y - self.top_left_y

    @property
    def top_left_xy(self) -> tuple[int, int]:
        """Retrieve the (x,y) coordinates of the top-left corner of the bounding box."""
        return (self.top_left_x, self.top_left_y)

    @property
    def bottom_right_xy(self) -> tuple[int, int]:
        """Retrieve the (x,y) coordinates of the bottom-right corner of the bounding box."""
        return (self.bottom_right_x, self.bottom_right_y)

    def get_center_xy(self) -> tuple[int, int]:
        """Compute the center pixel of the bounding box as an (x,y) coordinate."""
        center_x = int(self.top_left_x + 0.5 * self.width)
        center_y = int(self.top_left_y + 0.5 * self.height)
        return (center_x, center_y)

    def draw(self, image: np.ndarray, color: RGB = (255, 255, 0), thickness: int = 3) -> None:
        """Draw the bounding box as a rectangle on the given image.

        :param image: Image on which the bounding box is drawn (modified in-place)
        :param color: RGB color of the drawn bounding box
        :param thickness: Thickness (pixels) of the drawn bounding box
        """
        cv2.rectangle(image, self.top_left_xy, self.bottom_right_xy, color, thickness)
