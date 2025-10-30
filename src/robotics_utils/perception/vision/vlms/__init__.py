"""Import classes providing interfaces for VLMs."""

from .bounding_box_detector import ObjectBoundingBox as ObjectBoundingBox
from .bounding_box_detector import ObjectBoundingBoxes as ObjectBoundingBoxes
from .keypoint_detector import KeypointDetector as KeypointDetector
from .keypoint_detector import ObjectKeypoint as ObjectKeypoint
from .keypoint_detector import ObjectKeypoints as ObjectKeypoints
from .owl_vit_bounding_box_detector import OwlViTBoundingBoxDetector as OwlViTBoundingBoxDetector
from .queries import TextQueries as TextQueries
