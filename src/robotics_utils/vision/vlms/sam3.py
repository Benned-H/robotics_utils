"""Define a class providing object segmentation using Segment Anything 3 (SAM 3)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

from robotics_utils.vision import BoundingBox, PixelXY, RGBImage, determine_pytorch_device
from robotics_utils.vision.vlms.segmentation import InstanceSegmentation, ObjectSegmentations


class SAM3:
    """Segment objects in images based on text prompts."""

    def __init__(self) -> None:
        """Initialize a SAM 3 model for object segmentation."""
        self.device = determine_pytorch_device()
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def segment(self, image: RGBImage, queries: list[str]) -> ObjectSegmentations:
        """Segment an image using the given text prompts.

        :param image: Image to be segmented
        :param queries: Text queries describing the object(s) to be segmented
        :return: Collection of object segmentations detected in the image
        """
        image_inputs = self.processor(images=image.data, return_tensors="pt").to(self.device)

        # Pre-compute vision embeddings
        # Reference: https://github.com/huggingface/transformers/issues/42375#issuecomment-3576528458
        vision_embeds = self.model.get_vision_features(pixel_values=image_inputs.pixel_values)

        segmentations = []

        for query in queries:
            text_inputs = self.processor(text=query, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)

            # TODO: What do the following hyperparams mean/do?
            # TODO: Should I instead use processor.post_process_masks for efficiency?
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=image_inputs.get("original_sizes").tolist(),
            )[0]  # TODO: Why indexed 0?

            # Let N = the number of segmentations found for the query
            masks = results["masks"].numpy(force=True)  # (N, H, W)
            boxes = results["boxes"].tolist()  # (N, 4)
            scores = results["scores"].tolist()  # (N,)

            segmentations.extend(
                InstanceSegmentation(
                    query=query,
                    mask=masks[i, :, :].astype(np.bool),
                    bbox=BoundingBox(PixelXY(boxes[i][:2]), PixelXY(boxes[i][2:])),
                    score=score,
                )
                for i, score in enumerate(scores)
            )

        return ObjectSegmentations(segmentations=segmentations, image=image)
