"""Define a class to interface with an Intel RealSense camera."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import numpy as np
import pyrealsense2 as rs2
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table
from typing_extensions import Self

from robotics_utils.filesystem.logging import log_info
from robotics_utils.sensors.cameras import CameraIntrinsics, DepthCameraSpec, Resolution
from robotics_utils.vision.images import DepthImage, RGBDImage, RGBImage


@dataclass(order=True, frozen=True)
class CoreProfile:
    """Immutable ID for a RealSense stream profile, with natural ordering."""

    name: str  # Human-readable name
    index: int
    uid: int
    stream_type: str

    def __str__(self) -> str:
        """Return a human-readable representation of the core stream profile."""
        return f"{self.name}/{self.index}/{self.uid}/{self.stream_type}"

    @classmethod
    def from_profile(cls, stream: rs2.stream_profile) -> CoreProfile:
        """Construct a CoreProfile from a RealSense stream profile."""
        return CoreProfile(
            name=str(stream.stream_name()),
            index=int(stream.stream_index()),
            uid=int(stream.unique_id()),
            stream_type=str(stream.stream_type()).lower(),
        )


@dataclass(frozen=True)
class StreamVariant:
    """A variant of a RealSense stream profile."""

    core: CoreProfile
    resolution: Resolution | None
    fmt: str
    fps: int
    is_default: bool

    @classmethod
    def from_profile(cls, stream: rs2.stream_profile) -> StreamVariant:
        """Construct a StreamVariant from a RealSense stream profile."""
        resolution = None
        if stream.is_video_stream_profile():
            vsp = stream.as_video_stream_profile()
            resolution = Resolution(width=vsp.width(), height=vsp.height())

        return StreamVariant(
            core=CoreProfile.from_profile(stream),
            fmt=str(stream.format()).lower(),
            fps=int(stream.fps()),
            resolution=resolution,
            is_default=bool(stream.is_default()),
        )


def get_all_stream_variants() -> list[StreamVariant]:
    """Query every sensor and return a flat list of all available stream variants."""
    return [
        StreamVariant.from_profile(profile)
        for sensor in rs2.context().query_all_sensors()
        for profile in sensor.get_stream_profiles()
    ]


def display_streams_table(variants: list[StreamVariant]) -> None:
    """Render a table of all available stream profiles, marking default streams in green."""
    core_res_fmt_to_fps: dict[tuple[CoreProfile, Resolution | None, str], set[int]] = {}
    default_variants: set[tuple[CoreProfile, Resolution | None, str, int]] = set()

    for v in variants:
        key1 = (v.core, v.resolution, v.fmt)
        core_res_fmt_to_fps.setdefault(key1, set()).add(v.fps)

        if v.is_default:
            default_variants.add((v.core, v.resolution, v.fmt, v.fps))

    # Group rows of differing formats if they share a common set of possible FPS
    core_res_fps_to_fmts: dict[tuple[CoreProfile, Resolution | None, frozenset[int]], set[str]] = {}
    for (core, res, fmt), fps_set in core_res_fmt_to_fps.items():
        frozen_fps_set: frozenset[int] = frozenset(fps_set)
        key2 = (core, res, frozen_fps_set)
        core_res_fps_to_fmts.setdefault(key2, set()).add(fmt)

    rows = []
    for (core, res, frozen_fps_set), formats in core_res_fps_to_fmts.items():
        default_fps_set: set[int] = set()  # Set of FPS that appear in a default stream
        default_fmt_set: set[str] = set()  # Set of formats that appear in a default stream

        for fmt in formats:
            for fps in frozen_fps_set:
                if (core, res, fmt, fps) in default_variants:
                    default_fps_set.add(fps)
                    default_fmt_set.add(fmt)

        # Build the FPS and format cells by using bold green for default stream settings
        fps_cells = []
        for fps in sorted(frozen_fps_set):
            fps_text = f"[bold green]{fps}[/]" if (fps in default_fps_set) else str(fps)
            fps_cells.append(fps_text)
        fps_cell = ", ".join(fps_cells)

        fmt_cells = []
        for fmt in sorted(formats):
            fmt_text = f"[bold green]{fmt}[/]" if (fmt in default_fmt_set) else fmt
            fmt_cells.append(fmt_text)
        fmt_cell = ", ".join(fmt_cells)

        default_row = bool(default_fps_set or default_fmt_set)
        rows.append((core, res, fmt_cell, fps_cell, default_row))

    rows.sort(key=lambda r: (r[0], r[1]))  # Sort by core profile, then resolution

    table = Table(title="Available RealSense Streams")
    table.add_column("Stream (name/index/UID/type)", style="cyan", no_wrap=True)
    table.add_column("Resolution", style="yellow")
    table.add_column("Format", style="magenta")
    table.add_column("FPS", style="white")

    for core, resolution, fmt_cell, fps_cell, default_row in rows:
        core_cell = f"[bold green]{core}[/]" if default_row else str(core)
        res_str = str(resolution) if resolution is not None else "-"
        res_cell = f"[bold green]{res_str}[/]" if default_row else res_str
        table.add_row(core_cell, res_cell, fmt_cell, fps_cell)

    Console().print(table)


def expected_np_dtype(fmt: str) -> np.typing.DTypeLike | None:
    """Find the expected NumPy datatype for a RealSense stream format (or None if unspecified)."""
    if fmt == "format.z16":
        return np.uint16
    if fmt == "format.rgb8":
        return np.uint8
    if fmt == "format.motion_xyz32f":
        return np.uint8

    log_info(f"RealSense stream format '{fmt}' has no NumPy datatype specified...")
    return None


class RealSense:
    """A wrapper class for an Intel RealSense camera."""

    def __init__(self, depth_spec: DepthCameraSpec) -> None:
        """Initialize a pipeline to communicate with RealSense devices."""
        self.depth_spec = depth_spec

        self.pipeline = rs2.pipeline()
        self.profile: rs2.pipeline_profile | None = None

        self.depth_sensor: rs2.depth_sensor | None = None
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

        # Log information about all available RealSense streams
        display_streams_table(variants=get_all_stream_variants())

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

    def _process_frame(self, frame: rs2.frame) -> None:
        """Process the given pyrealsense2.frame from the RealSense camera."""
        stream_info = StreamVariant.from_profile(frame.get_profile())

        frame_data = np.asanyarray(frame.data)
        expected_dtype = expected_np_dtype(stream_info.fmt)
        if frame_data.dtype != expected_dtype:
            raise TypeError(f"Expected NumPy datatype {expected_dtype}, got {frame_data.dtype}")

        if stream_info.core.stream_type == "stream.color":
            self._latest_rgb = RGBImage(frame_data)
            return

        if stream_info.core.stream_type == "stream.depth":
            depth_m = frame_data * self.depth_scale_to_m

            # Zero out any depth values outside the camera's operating range
            depth_m[depth_m < self.depth_spec.min_range_m] = 0
            depth_m[depth_m > self.depth_spec.max_range_m] = 0

            self._latest_depth = DepthImage(depth_m)
            return

        if stream_info.core.stream_type == "stream.gyro":
            self._latest_gyro = frame_data
            return

        if stream_info.core.stream_type == "stream.accel":
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
