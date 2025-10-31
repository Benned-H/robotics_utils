"""Import classes providing interfaces for VLMs."""

from .bounding_box_detection import DetectedBoundingBox as DetectedBoundingBox
from .bounding_box_detection import DetectedBoundingBoxes as DetectedBoundingBoxes
from .keypoint_detection import DetectedKeypoint as DetectedKeypoint
from .keypoint_detection import DetectedKeypoints as DetectedKeypoints
from .keypoint_detection import KeypointDetector as KeypointDetector
from .owl_vit_bounding_box_detector import OwlViTBoundingBoxDetector as OwlViTBoundingBoxDetector
from .queries import TextQueries as TextQueries
