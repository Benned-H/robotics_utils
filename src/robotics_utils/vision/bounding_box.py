"""Define a dataclass to represent a bounding box in an image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from robotics_utils.vision.rgb_image import RGBImage
from robotics_utils.vision.vision_utils import RGB


class PixelXY:
    """An (x,y) coordinate of a pixel in an image."""

    def __init__(self, xy: tuple[int, int] | NDArray) -> None:
        """Initialize the PixelXY using the given (x,y) coordinate values."""
        if isinstance(xy, tuple):
            xy = np.array(xy)

        self.xy = xy.astype(int)

    def __add__(self, other: PixelXY) -> PixelXY:
        """Find the sum of this PixelXY and another."""
        return PixelXY(self.xy + other.xy)

    def __mul__(self, value: float) -> PixelXY:
        """Find the product of this PixelXY and the given scalar."""
        return PixelXY(self.xy * value)

    @property
    def x(self) -> int:
        """Retrieve the x-coordinate of this pixel."""
        return self.xy[0]

    @property
    def y(self) -> int:
        """Retrieve the y-coordinate of this pixel."""
        return self.xy[1]

    def to_tuple(self) -> tuple[int, int]:
        """Convert the PixelXY into an (x,y) tuple."""
        return tuple(self.xy)


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular bounding box in an image."""

    top_left: PixelXY
    bottom_right: PixelXY

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
        """Construct a bounding box from a center (x,y) pixel, a width, and a height.

        :param center_pixel: Center pixel of the bounding box as an (x,y) image coordinate
        :param height: Height of the bounding box (in pixels)
        :param width: Width of the bounding box (in pixels)
        :return: Constructed BoundingBox instance
        """
        half_size = 0.5 * np.array([width, height])

        top_left = np.ceil(center_pixel.xy - half_size)
        bottom_right = np.floor(center_pixel.xy + half_size)

        return BoundingBox(PixelXY(top_left), PixelXY(bottom_right))  # TODO: Test these dimensions

    @property
    def width(self) -> int:
        """Compute the width (in pixels) of the bounding box."""
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self) -> int:
        """Compute the height (in pixels) of the bounding box."""
        return self.bottom_right.y - self.top_left.y

    @property
    def center_xy(self) -> PixelXY:
        """Compute the center pixel of the bounding box as an (x,y) coordinate."""
        return (self.top_left + self.bottom_right) * 0.5

    def draw(self, image: RGBImage, color: RGB, thickness: int = 3) -> None:
        """Draw the bounding box as a rectangle on the given image.

        :param image: Image on which the bounding box is drawn (modified in-place)
        :param color: RGB color of the drawn bounding box
        :param thickness: Thickness (pixels) of the drawn bounding box
        """
        cv2.rectangle(
            image.data,
            self.top_left.to_tuple(),
            self.bottom_right.to_tuple(),
            color,
            thickness,
        )
        cv2.circle(image.data, self.center_xy.to_tuple(), 1, color, thickness)

    def crop(self, image: RGBImage, scale_box: float = 1.0) -> RGBImage:
        """Return a crop of the given image based on this bounding box.

        :param image: RGB image from which a cropped image is created
        :param scale_box: Ratio to scale the bounding box size (defaults to 1.0)
        :return: New RGB image containing the cropped section of the given image
        """
        scaled_height = int(self.height * scale_box)
        scaled_width = int(self.width * scale_box)
        scaled_box = BoundingBox.from_center(self.center_xy, scaled_height, scaled_width)

        min_x, min_y = scaled_box.top_left.to_tuple()
        max_x, max_y = scaled_box.bottom_right.to_tuple()

        cropped_data = image.data[min_y:max_y, min_x:max_x, :]  # TODO: Ensure is within the image
        return RGBImage(cropped_data)
