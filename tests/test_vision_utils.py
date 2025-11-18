"""Unit tests for computer vision utility functions defined in vision_utils.py."""

import torch

from robotics_utils.io import log_info
from robotics_utils.math.conversions import Gi_TO_B
from robotics_utils.vision import determine_pytorch_device


def test_determine_pytorch_device() -> None:
    """Verify that a PyTorch device can be successfully identified for the local machine."""
    # Arrange/Act - Call the function to determine the appropriate local PyTorch device
    result_device = determine_pytorch_device()
    log_info(f"Selected PyTorch device: {result_device}")

    # Assert - Expect that a CUDA device can provide a memory estimate
    if result_device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=result_device)
        log_info(f"Free CUDA memory: {free_bytes / Gi_TO_B:.3f} GiB")
        log_info(f"Total CUDA memory: {total_bytes / Gi_TO_B:.3f} GiB")
