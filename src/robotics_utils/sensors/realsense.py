"""Define a class to interface with an Intel RealSense camera."""

from __future__ import annotations

from types import TracebackType

import numpy as np
import pyrealsense2 as rs
from typing_extensions import Self

from robotics_utils.sensors.cameras import CameraIntrinsics, DepthCameraSpec
from robotics_utils.vision.images import DepthImage, RGBDImage, RGBImage


class RealSense:
    """A wrapper class for an Intel RealSense camera."""

    def __init__(self, depth_spec: DepthCameraSpec) -> None:
        """Initialize a pipeline to communicate with RealSense devices."""
        self.depth_spec = depth_spec
        self.pipeline = rs.pipeline()
        self.profile: rs.pipeline_profile | None = None

        self.depth_sensor: rs.depth_sensor | None = None
        self.depth_scale_to_m: float | None = None
        """Scale between units of the depth image and meters."""

        self._rgb_intrinsics: CameraIntrinsics | None = None
        self._depth_intrinsics: CameraIntrinsics | None = None

    def __enter__(self) -> Self:
        """Enter a managed context for streaming data from an Intel RealSense."""
        self.profile = self.pipeline.start()

        self.rgb_sensor = self.profile.get_device().first_color_sensor()

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale_to_m = self.depth_sensor.get_depth_scale()

        self._populate_intrinsics()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Exit a managed context for streaming data from an Intel RealSense.

        :param exc_type: Type of exception raised in the context (None if no exception)
        :param exc_value: Value of the exception raised in the context (None if no exception)
        :param traceback: Traceback of the exception raised in the context (None if no exception)
        :return: True if exception is suppressed, False if exception should propagate, else None
        """
        self.pipeline.stop()
        return None

    @property
    def rgb_intrinsics(self) -> CameraIntrinsics:
        """Retrieve the RealSense's RGB camera intrinsics."""
        if self._rgb_intrinsics is None:
            raise RuntimeError("RGB camera intrinsics are unavailable")
        return self._rgb_intrinsics

    @property
    def depth_intrinsics(self) -> CameraIntrinsics:
        """Retrieve the RealSense's depth camera intrinsics."""
        if self._depth_intrinsics is None:
            raise RuntimeError("Depth camera intrinsics are unavailable")
        return self._depth_intrinsics

    def get_rgbd(self, timeout_ms: int = 500) -> RGBDImage:
        """Wait for an RGB-D image from the RealSense pipeline.

        :param timeout_ms: Timeout for frame waiting, defaults to 500 ms (0.5 s)
        :return: RGB-D image read from the RealSense
        """
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

        color_data = np.asanyarray(frames.get_color_frame().data)

        depth_data = np.asanyarray(frames.get_depth_frame().data)
        depth_m = depth_data * self.depth_scale_to_m

        # Zero out any depth values outside the camera's operating range
        depth_m[depth_m < self.depth_spec.min_range_m] = 0
        depth_m[depth_m > self.depth_spec.max_range_m] = 0

        return RGBDImage(rgb=RGBImage(color_data), depth=DepthImage(depth_m))

    def _populate_intrinsics(self) -> None:
        """Retrieve the camera intrinsics from the current RealSense device, if available."""
        if self.profile is None:
            return

        video_streams = [
            stream.as_video_stream_profile()
            for stream in self.profile.get_streams()
            if stream.is_video_stream_profile
        ]

        for vs in video_streams:
            stream_name = vs.stream_name().lower()
            rs_i = vs.get_intrinsics()
            intrinsics = CameraIntrinsics(rs_i.fx, rs_i.fy, rs_i.ppx, rs_i.ppy)

            if stream_name == "color":
                self._rgb_intrinsics = intrinsics
            elif stream_name == "depth":
                self._depth_intrinsics = intrinsics
