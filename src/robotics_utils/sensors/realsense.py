"""Define a class to interface with an Intel RealSense camera."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType

import numpy as np
import pyrealsense2 as rs
from numpy.typing import NDArray
from typing_extensions import Self

from robotics_utils.filesystem.logging import log_info
from robotics_utils.sensors.cameras import CameraIntrinsics, DepthCameraSpec
from robotics_utils.vision.images import DepthImage, RGBDImage, RGBImage


class StreamType(Enum):
    """An enumeration of stream types available from Intel RealSense devices."""

    RGB = 0
    DEPTH = 1
    GYRO = 2
    ACCEL = 3

    @classmethod
    def from_string(cls, string: str) -> StreamType:
        """Construct a StreamType corresponding to the given string."""
        string = str(string).lower().strip()

        if "color" in string:
            return StreamType.RGB

        if "depth" in string:
            return StreamType.DEPTH

        if "gyro" in string:
            return StreamType.GYRO

        if "accel" in string:
            return StreamType.ACCEL

        raise ValueError(f"Unknown type of RealSense stream: '{string}'")


def string_to_np_dtype(string: str) -> np.typing.DTypeLike:
    """Map the given string to the corresponding NumPy datatype."""
    if string == "format.z16":
        return np.uint16
    if string == "format.rgb8":
        return np.uint8
    if string == "format.motion_xyz32f":
        return np.uint8

    raise ValueError(f"Unexpected RealSense data format: '{string}'")


@dataclass(frozen=True)
class StreamInfo:
    """Data characterizing a stream of data from an Intel RealSense."""

    name: str
    uid: int
    type_: StreamType
    dtype: np.typing.DTypeLike

    @classmethod
    def from_stream_profile(cls, stream: rs.stream_profile) -> StreamInfo:
        """Construct a StreamInfo from a RealSense stream profile."""
        return StreamInfo(
            name=str(stream.stream_name()),
            uid=int(stream.unique_id()),
            type_=StreamType.from_string(stream.stream_type()),
            dtype=string_to_np_dtype(str(stream.format())),
        )


class RealSense:
    """A wrapper class for an Intel RealSense camera."""

    def __init__(self, depth_spec: DepthCameraSpec) -> None:
        """Initialize a pipeline to communicate with RealSense devices."""
        self.depth_spec = depth_spec
        self.pipeline = rs.pipeline()
        self.profile: rs.pipeline_profile | None = None
        self.streams: dict[str, StreamInfo] = {}  # Maps each stream name to its identifying info

        self.depth_sensor: rs.depth_sensor | None = None
        self.depth_scale_to_m: float | None = None
        """Scale between units of the depth image and meters."""

        self._rgb_intrinsics: CameraIntrinsics | None = None
        self._depth_intrinsics: CameraIntrinsics | None = None

        self._latest_rgb: RGBImage | None = None
        self._latest_depth: DepthImage | None = None
        self._latest_gyro: NDArray[np.uint8] | None = None
        self._latest_accel: NDArray[np.uint8] | None = None

    def __enter__(self) -> Self:
        """Enter a managed context for streaming data from an Intel RealSense."""
        self.profile = self.pipeline.start()

        # Log information about the available streams from the RealSense device
        for stream in self.profile.get_streams():
            stream_info = StreamInfo.from_stream_profile(stream)
            self.streams[stream_info.name] = stream_info
            log_info(f"Stream {len(self.streams)}: {stream_info.name}")

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

    def _update(self, timeout_ms: int = 500) -> None:
        """Update the stored data with the latest data from the RealSense device."""
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        frames.foreach(self._process_frame)

    def _process_frame(self, frame: rs.frame) -> None:
        """Process the given pyrealsense2.frame from the RealSense camera."""
        frame_info = StreamInfo.from_stream_profile(frame.get_profile())
        frame_data = np.asanyarray(frame.data)
        if frame_data.dtype != frame_info.dtype:
            raise TypeError(f"Expected NumPy datatype {frame_info.dtype}, got {frame_data.dtype}")

        if frame_info.type_ == StreamType.RGB:
            self._latest_rgb = RGBImage(frame_data)
            return

        if frame_info.type_ == StreamType.DEPTH:
            depth_m = frame_data * self.depth_scale_to_m

            # Zero out any depth values outside the camera's operating range
            depth_m[depth_m < self.depth_spec.min_range_m] = 0
            depth_m[depth_m > self.depth_spec.max_range_m] = 0

            self._latest_depth = DepthImage(depth_m)
            return

        if frame_info.type_ == StreamType.GYRO:
            self._latest_gyro = frame_data
            return

        if frame_info.type_ == StreamType.ACCEL:
            self._latest_accel = frame_data
            return

        raise ValueError(f"Unrecognized type of RealSense frame: {frame}")

    def get_rgbd(self, timeout_ms: int = 500) -> RGBDImage:
        """Wait for an RGB-D image from the RealSense pipeline.

        :param timeout_ms: Timeout for frame waiting, defaults to 500 ms (0.5 s)
        :return: RGB-D image read from the RealSense
        """
        self._update(timeout_ms=timeout_ms)
        if self._latest_rgb is None or self._latest_depth is None:
            raise RuntimeError("Unable to retrieve RGB-D data from the RealSense device.")

        return RGBDImage(self._latest_rgb, self._latest_depth)

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
            if not vs:
                continue
            stream_name = vs.stream_name()  # 'Color' or 'Depth'
            rs_i = vs.get_intrinsics()
            intrinsics = CameraIntrinsics(rs_i.fx, rs_i.fy, rs_i.ppx, rs_i.ppy)

            if stream_name == "Color":
                self._rgb_intrinsics = intrinsics
            elif stream_name == "Depth":
                self._depth_intrinsics = intrinsics
