"""Define a dataclass to represent a cropped patch of an image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.vision.image_processing.bounding_box import BoundingBox
from robotics_utils.vision.image_processing.image import ImageT

if TYPE_CHECKING:
    from robotics_utils.vision.image_processing.pixel_xy import PixelXY


@dataclass(frozen=True)
class ImagePatch(Generic[ImageT]):
    """A cropped patch of an image with its associated bounding box."""

    patch: ImageT
    bounding_box: BoundingBox

    @classmethod
    def crop(cls, image: ImageT, bounding_box: BoundingBox) -> ImagePatch[ImageT]:
        """Crop the given image based on the given bounding box.

        :param image: Image from which the crop is taken
        :param bounding_box: Bounding box specifying the crop shape
        :return: Cropped image patch and the bounding box (clipped into the image, if necessary)
        """
        min_p = image.clip_pixel(bounding_box.top_left)
        max_p = image.clip_pixel(bounding_box.bottom_right)
        clipped_box = BoundingBox(min_p, max_p)

        cropped_data = image.data[min_p.y : max_p.y + 1, min_p.x : max_p.x + 1, ...]
        cropped_image = type(image)(cropped_data.copy())

        return ImagePatch(patch=cropped_image, bounding_box=clipped_box)

    @classmethod
    def scaled_crop(cls, image: ImageT, box: BoundingBox, scaling: float) -> ImagePatch[ImageT]:
        """Crop an image based on a scaled version of a bounding box.

        :param image: Image from which the crop is taken
        :param box: Bounding box specifying the crop shape
        :param scaling: Ratio to scale the bounding box size (e.g., 1.0 produces the same size)
        :return: Cropped image patch and the bounding box (scaled and clipped, if necessary)
        """
        scaled_w = int(scaling * box.width)
        scaled_h = int(scaling * box.height)
        scaled_box = BoundingBox.from_center(box.center_pixel, width=scaled_w, height=scaled_h)

        return ImagePatch.crop(image=image, bounding_box=scaled_box)

    @classmethod
    def crop_around_pixel(cls, image: ImageT, pixel: PixelXY, patch_size: int = 100) -> ImagePatch:
        """Crop a square patch around the given pixel in the given image.

        :param image: Image from which the patch is cropped
        :param pixel: (x,y) pixel coordinate at the center of the patch
        :param patch_size: Size of the square patch (side length in pixels)
        :return: Image patch including cropped image data and clipped bounding box
        """
        bounding_box = BoundingBox.from_center(pixel, height=patch_size, width=patch_size)
        return ImagePatch.crop(image=image, bounding_box=bounding_box)
