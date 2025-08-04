"""Define dataclasses to represent images and related concepts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from robotics_utils.visualization import Displayable


class Image(ABC, Displayable):
    """An image represented as a NumPy array."""

    @abstractmethod
    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the image data into a form that can be visualized."""

    def __init__(self, data: NDArray) -> None:
        """Initialize the image using the given array."""
        if len(data.shape) < 2:
            raise ValueError(f"Image expects at least 2-dim. data, got {data.shape}")

        self.data = data

    @property
    def height(self) -> int:
        """Retrieve the height (in pixels) of the image."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Retrieve the width (in pixels) of the image."""
        return self.data.shape[1]

    @property
    def height_width(self) -> tuple[int, int]:
        """Retrieve a tuple containing the height and width (in pixels) of the image."""
        return (self.height, self.width)

    def clip_x(self, pixel_x: int) -> int:
        """Clip a pixel x-coordinate into the image."""
        return np.clip(pixel_x, a_min=0, a_max=self.width - 1)

    def clip_y(self, pixel_y: int) -> int:
        """Clip a pixel y-coordinate into the image."""
        return np.clip(pixel_y, a_min=0, a_max=self.height - 1)

    def clip_pixel(self, pixel_xy: PixelXY) -> PixelXY:
        """Clip the given (x,y) coordinate of a pixel into the image."""
        return PixelXY((self.clip_x(pixel_xy.x), self.clip_y(pixel_xy.y)))

    def get_crop(self, top_left: PixelXY, bottom_right: PixelXY) -> Self:
        """Retrieve the specified crop of the image.

        :param top_left: (x,y) coordinates of the top-left pixel in the cropped image
        :param bottom_right: (x,y) coordinates of the bottom-right pixel in the cropped image
        :return: New image containing the cropped portion
        """
        min_x, min_y = self.clip_pixel(top_left)
        max_x, max_y = self.clip_pixel(bottom_right)

        cropped_data = self.data[min_y : max_y + 1, min_x : max_x + 1, :]
        return type(self)(cropped_data.copy())


class RGBImage(Image):
    """An RGB image represented as a NumPy array of shape (H, W, 3)."""

    def __init__(self, data: NDArray) -> None:
        """Initialize the RGB image using the given array of data."""
        super().__init__(data)

        # Verify expected properties of RGB image data
        if len(self.data.shape) != 3:
            raise ValueError(f"RGBImage expects 3-dim. data, got {self.data.shape}")

        if self.data.shape[2] != 3:
            raise ValueError(f"RGBImage expects 3 channels of color, got {self.data.shape[2]}")

        if self.data.dtype != np.uint8:
            raise TypeError(f"RGBImage expects datatype np.uint8, got {self.data.dtype}")

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the RGB image into a form that can be visualized."""
        color_converted = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        return color_converted.astype(np.uint8)

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


class DepthImage(Image):
    """A depth image represented as a NumPy array of shape (H, W)."""

    def __init__(self, data: NDArray) -> None:
        """Initialize the depth image using an array of shape (H, W) of depth data (in meters)."""
        super().__init__(data)

        # Verify expected properties of depth image data
        if len(self.data.shape) != 2:
            raise ValueError(f"DepthImage expects 2-dim. data, got {self.data.shape}")

        if self.data.dtype != np.float64:  # Expect floats because the depth data is in meters
            raise TypeError(f"DepthImage expects datatype np.float64, got {self.data.dtype}")

    @property
    def min_depth_m(self) -> float:
        """Retrieve the minimum non-zero depth (meters) in the image."""
        return np.min(self.data[self.data > 0])

    @property
    def max_depth_m(self) -> float:
        """Retrieve the maximum depth (meters) in the image."""
        return np.max(self.data)

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the depth image into a form that can be visualized."""
        normalized_depth = cv2.normalize(self.data, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        convert_to_uint8 = np.astype(normalized_depth, np.uint8)
        return cv2.applyColorMap(convert_to_uint8, colormap=cv2.COLORMAP_JET)


@dataclass
class RGBDImage(Displayable):
    """An RGB-D image represented using a pair of RGB and depth images."""

    rgb: RGBImage
    depth: DepthImage

    def __post_init__(self) -> None:
        """Verify that the constructed RGBDImage is valid."""
        if self.rgb.height_width != self.depth.height_width:
            raise ValueError(
                "Invalid RGB-D image dimensions.\n\t"
                f"RGB (H, W): {self.rgb.height_width} Depth (H, W): {self.depth.height_width}",
            )

    @property
    def height(self) -> int:
        """Retrieve the height of the RGB-D image (identical for the RGB and depth parts)."""
        return self.rgb.height

    @property
    def width(self) -> int:
        """Retrieve the width of the RGB-D image (identical for the RGB and depth parts)."""
        return self.rgb.width

    @property
    def height_width(self) -> tuple[int, int]:
        """Retrieve a tuple containing the height and width (in pixels) of the RGB-D image."""
        return (self.height, self.width)

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the RGBDImage into a form that can be visualized."""
        rgb_viz = self.rgb.convert_for_visualization()
        depth_viz = self.depth.convert_for_visualization()
        return np.concatenate((rgb_viz, depth_viz), axis=0)  # Axis 0 = Height (vertical)


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
