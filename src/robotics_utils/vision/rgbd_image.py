"""Define a class to represent RGB-D (i.e., RGB color + depth) images."""

from dataclasses import dataclass

import numpy as np

from robotics_utils.vision.depth_image import DepthImage
from robotics_utils.vision.rgb_image import RGBImage
from robotics_utils.visualization.display_images import Displayable


@dataclass
class RGBDImage(Displayable):
    """An RGB-D image represented using a pair of RGB and depth images."""

    rgb: RGBImage
    depth: DepthImage

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Convert the RGBDImage into a form that can be visualized."""
        rgb_viz = self.rgb.convert_for_visualization()
        depth_viz = self.depth.convert_for_visualization()

        max_width = max(self.rgb.width, self.depth.width)
        rgb_stack = np.zeros(shape=(self.rgb.height, max_width, 3), dtype=np.uint8)
        rgb_stack[:, 0 : self.rgb.width, :] = rgb_viz
        depth_stack = np.zeros(shape=(self.depth.height, max_width, 3), dtype=np.uint8)
        depth_stack[:, 0 : self.depth.width, :] = depth_viz

        return np.concatenate((rgb_stack, depth_stack), axis=0)  # Axis 0 = Height (vertical)

    @property
    def same_dimensions(self) -> bool:
        """Check whether the RGB and depth images have the same width and height (True = same)."""
        return (self.rgb.width == self.depth.width) and (self.rgb.height == self.depth.height)
