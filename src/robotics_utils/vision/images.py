"""Define dataclasses to represent images and related concepts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


class RGBImage:
    """An RGB image represented as a NumPy array."""

    def __init__(self, data: NDArray) -> None:
        """Initialize the RGB image using the given array of data."""
        if data.dtype != np.uint8:
            data = (data * 255).astype(np.uint8)

        self.data = data

        ### Verify expected properties of the RGB image data ###
        if len(self.data.shape) != 3:
            raise ValueError(f"RGBImage expects 3-dimensional data, got {self.data.shape}")

        if self.data.shape[2] != 3:
            raise ValueError(f"RGBImage expects 3 channels of color, got {self.data.shape[2]}")

        if self.data.dtype != np.uint8:
            raise TypeError(f"RGBImage expects datatype np.uint8, got {self.data.dtype}")

    @classmethod
    def from_file(cls, image_path: str | Path) -> RGBImage:
        """Load an RGB image from the given filepath."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Cannot load image from nonexistent file: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image from path: {image_path}")

        rgb_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return RGBImage(rgb_data)

    @property
    def height(self) -> int:
        """Retrieve the height (in pixels) of the image."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Retrieve the width (in pixels) of the image."""
        return self.data.shape[1]

    def visualize(self, window_title: str) -> None:
        """Visualize the image in an OpenCV2 window with the given title."""
        image_bgr = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def clip_x(self, pixel_x: int) -> int:
        """Clip a pixel x-coordinate into the image."""
        return np.clip(pixel_x, a_min=0, a_max=self.width - 1)

    def clip_y(self, pixel_y: int) -> int:
        """Clip a pixel y-coordinate into the image."""
        return np.clip(pixel_y, a_min=0, a_max=self.height - 1)

    def clip_pixel(self, pixel_xy: PixelXY) -> PixelXY:
        """Clip the given (x,y) coordinate of a pixel into the image."""
        return PixelXY((self.clip_x(pixel_xy.x), self.clip_y(pixel_xy.y)))

    def get_crop(self, top_left: PixelXY, bottom_right: PixelXY) -> RGBImage:
        """Retrieve the specified crop of the RGB image.

        :param top_left: (x,y) coordinates of the top-left pixel in the cropped image
        :param bottom_right: (x,y) coordinates of the bottom-right pixel in the cropped image
        :return: New RGBImage containing the cropped portion of the image
        """
        min_x, min_y = self.clip_pixel(top_left)
        max_x, max_y = self.clip_pixel(bottom_right)

        cropped_data = self.data[min_y : max_y + 1, min_x : max_x + 1, :]
        return RGBImage(cropped_data.copy())


class PixelXY:
    """An (x,y) coordinate of a pixel in an image."""

    def __init__(self, xy: tuple[int, int] | NDArray) -> None:
        """Initialize the PixelXY using the given (x,y) coordinate values."""
        if isinstance(xy, tuple):
            xy = np.array(xy)

        self.xy = xy.astype(np.int_)

    def __add__(self, other: PixelXY) -> PixelXY:
        """Find the sum of this PixelXY and another."""
        return PixelXY(self.xy + other.xy)

    def __iter__(self) -> Iterator[int]:
        """Provide an iterator over the (x,y) coordinates."""
        yield from self.xy

    def __str__(self) -> str:
        """Return a readable string representation of the pixel."""
        return f"({self.x}, {self.y})"

    @property
    def x(self) -> int:
        """Retrieve the x-coordinate of this pixel."""
        return self.xy[0]

    @property
    def y(self) -> int:
        """Retrieve the y-coordinate of this pixel."""
        return self.xy[1]

    def all_close(self, other: PixelXY) -> bool:
        """Evaluate whether this pixel is approximately equal to another pixel."""
        return np.allclose(self.xy, other.xy)
