"""Define a class providing object segmentation using Segment Anything 3 (SAM 3)."""

from __future__ import annotations

from importlib_metadata import version

try:
    from transformers import BatchEncoding, Sam3Model, Sam3Processor

    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False

import os

# Set PyTorch CUDA allocator to use expandable segments to reduce fragmentation
# Must be set before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from robotics_utils.vision import BoundingBox, PixelXY, RGBImage, determine_pytorch_device
from robotics_utils.vision.vlms.segmentation import ObjectSegmentation, ObjectSegmentations


class SAM3:
    """Segment objects in images based on text prompts."""

    def __init__(self) -> None:
        """Initialize a SAM 3 model for object segmentation."""
        if not SAM3_AVAILABLE:
            raise ImportError(
                f"Cannot import SAM 3 with 'transformers' version: {version('transformers')}",
            )

        self.device = determine_pytorch_device()

        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.model.eval()  # Set to eval mode to disable dropout and save memory

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self._query_cache: dict[str, BatchEncoding] = {}
        """Cache previously used text queries to save time transferring to the GPU."""

    def segment(
        self,
        image: RGBImage,
        queries: list[str],
        instance_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> ObjectSegmentations:
        """Segment an image using the given text prompts.

        :param image: Image to be segmented
        :param queries: Text queries describing the object(s) to be segmented
        :param instance_threshold: Probability score threshold to keep predicted instance masks
        :param mask_threshold: Threshold used to convert predicted masks into binary values
        :return: Collection of object segmentations detected in the image
        """
        # Clear GPU cache before processing to free memory from previous iterations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image_inputs = self.processor(images=image.data, return_tensors="pt").to(self.device)

        # Pre-compute vision embeddings
        # Reference: https://github.com/huggingface/transformers/issues/42375#issuecomment-3576528458
        vision_embeds = self.model.get_vision_features(pixel_values=image_inputs.pixel_values)
        original_sizes = image_inputs.get("original_sizes").tolist()

        del image_inputs  # Delete image_inputs to free up GPU memory (cache is cleared later)

        segmentations: list[ObjectSegmentation] = []

        for query in queries:
            # Use cached text inputs if available, otherwise process and cache
            if query not in self._query_cache:
                self._query_cache[query] = self.processor(text=query, return_tensors="pt").to(
                    self.device,
                )
            text_inputs = self._query_cache[query]

            with torch.no_grad():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)

            # Post-process the results to get instance segmentation masks
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=instance_threshold,
                mask_threshold=mask_threshold,
                target_sizes=original_sizes,
            )[0]  # Index 0 because we only provided one image

            # Let N = the number of segmentations found for the query
            masks = results["masks"].numpy(force=True)  # (N, H, W)
            boxes = results["boxes"].tolist()  # (N, 4)
            scores = results["scores"].tolist()  # (N,)

            segmentations.extend(
                ObjectSegmentation(
                    query=query,
                    mask=masks[i, :, :].astype(np.bool),
                    bbox=BoundingBox(PixelXY(boxes[i][:2]), PixelXY(boxes[i][2:])),
                    score=score,
                )
                for i, score in enumerate(scores)
            )

            del outputs, results  # Clean up outputs after processing each query to free GPU memory

        del vision_embeds  # Clean up vision embeddings before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ObjectSegmentations(segmentations=segmentations, image=image)
