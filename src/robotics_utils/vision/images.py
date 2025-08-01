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


def display_image(data: NDArray[np.uint8], window_title: str, wait_for_input: bool = True) -> bool:
    """Display the given image in a titled window.

    :param data: NumPy array containing image data to be displayed
    :param window_title: Title used for the display window
    :param wait_for_input: Whether to display the image until user input (defaults to True)
    :return: Boolean indicating if the window remains active (True = Active, False = Closed)
    """
    cv2.imshow(window_title, data)
    if wait_for_input:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False

    key_input = cv2.waitKey(1) & 0xFF
    if key_input == ord("q"):  # Close the window if the user inputs 'q'
        cv2.destroyAllWindows()
        return False

    return True


class Image(ABC):
    """An image represented as a NumPy array."""

    @abstractmethod
    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the image data for visualization in an OpenCV window.

        :return: Image data converted for visualization
        """

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

    def visualize(self, window_title: str, wait_for_input: bool = True) -> bool:
        """Visualize the image in an OpenCV2 window with the given title.

        :param window_title: Title used for the OpenCV window
        :param wait_for_input: Whether to display the image until user input (defaults to True)
        :return: Boolean indicating if the window remains active (True = Active, False = Closed)
        """
        visualize_data = self.convert_for_visualization()
        return display_image(visualize_data, window_title, wait_for_input)


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

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the depth image into a form that can be visualized."""
        normalized_depth = cv2.normalize(self.data, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        convert_to_uint8 = np.astype(normalized_depth, np.uint8)
        return cv2.applyColorMap(convert_to_uint8, colormap=cv2.COLORMAP_JET)


@dataclass
class RGBDImage:
    """An RGB-D image represented using a pair of RGB and depth images."""

    rgb: RGBImage
    depth: DepthImage

    def __post_init__(self) -> None:
        """Verify that the constructed RGBDImage is valid."""
        rgb_hw = (self.rgb.height, self.rgb.width)
        depth_hw = (self.depth.height, self.rgb.width)
        if rgb_hw != depth_hw:
            raise ValueError(f"Invalid RGB-D image dimensions: RGB: {rgb_hw} Depth: {depth_hw}")

    def visualize(self, window_title: str, wait_for_input: bool = True) -> bool:
        """Visualize the RGB-D image in an OpenCV2 window with the given title.

        :param window_title: Title used for the OpenCV window
        :param wait_for_input: Whether to display the image until user input (defaults to True)
        :return: Boolean indicating if the window remains active (True = Active, False = Closed)
        """
        rgb_viz = self.rgb.convert_for_visualization()
        depth_viz = self.depth.convert_for_visualization()
        stacked_images = np.concatenate((rgb_viz, depth_viz), axis=0)  # Axis 0 = Height (vertical)
        return display_image(stacked_images, window_title, wait_for_input)


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
