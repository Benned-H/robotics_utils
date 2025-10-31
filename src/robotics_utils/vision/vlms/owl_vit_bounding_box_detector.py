"""Define a class providing open-vocabulary object bounding box detection using OWL-ViT."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTImageProcessorFast, OwlViTProcessor

from robotics_utils.vision import (
    BoundingBox,
    PixelXY,
    RGBImage,
    determine_pytorch_device,
)
from robotics_utils.vision.vlms.bounding_box_detection import (
    BoundingBoxDetector,
    DetectedBoundingBox,
    DetectedBoundingBoxes,
)


class OwlViTBoundingBoxDetector(BoundingBoxDetector):
    """Detect object bounding boxes using the OWL-ViT model."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32") -> None:
        """Initialize the specified OWL-ViT model."""
        self.device = determine_pytorch_device()
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set the model into evaluation mode
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.fast_image_processor = OwlViTImageProcessorFast()

    def detect_bounding_boxes(self, image: RGBImage, queries: list[str]) -> DetectedBoundingBoxes:
        """Detect object bounding boxes matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of detected object bounding boxes matching the queries
        """
        pil_img = Image.fromarray(image.data).convert("RGB")

        # Process all text queries at once
        inputs = self.processor(text=queries, images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            raw_outputs = self.model(**inputs)  # Single forward pass

        # Post-process by providing the (H, W) per image
        results = self.processor.post_process_grounded_object_detection(
            outputs=raw_outputs,
            target_sizes=torch.tensor([pil_img.size[::-1]], device=self.device),
            threshold=0.1,
        )[0]  # Index 0 because we only provided one image

        scores = results["scores"].tolist()
        labels = results["labels"].tolist()  # Integer class labels (indices into `queries`)
        boxes = results["boxes"].tolist()  # (top left x, then y, bottom right x, then y)

        detections = [
            DetectedBoundingBox(
                queries[q_idx],
                BoundingBox(top_left=PixelXY(box_data[:2]), bottom_right=PixelXY(box_data[2:])),
                score,
            )
            for q_idx, box_data, score in zip(labels, boxes, scores)
        ]

        return DetectedBoundingBoxes(detections, image)
