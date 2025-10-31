"""Define a class to visualize pointclouds using Open3D."""

from __future__ import annotations

from typing import TYPE_CHECKING

import open3d as o3d

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from robotics_utils.vision import Pointcloud


class PointcloudVisualizer:
    """A manager context for live visualization of pointclouds using Open3D."""

    def __init__(self) -> None:
        """Initialize an Open3D visualizer for pointclouds."""
        self.vis = o3d.visualization.Visualizer()
        self.o3d_pcd = o3d.geometry.PointCloud()

        self.uncalled_while_active: bool | None = None
        """None = Visualizer inactive; True = Uncalled but live; False = Active and been called."""

    def __enter__(self) -> Self:
        """Enter a managed context for live pointcloud visualization."""
        self.vis.create_window()
        self.uncalled_while_active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Exit a managed context for live pointcloud visualization.

        :param exc_type: Type of exception raised in the context (None if no exception)
        :param exc_value: Value of the exception raised in the context (None if no exception)
        :param traceback: Traceback of the exception raised in the context (None if no exception)
        :return: True if exception is suppressed, False if exception should propagate, else None
        """
        self.vis.destroy_window()
        self.uncalled_while_active = None
        return None

    def visualize(self, pointcloud: Pointcloud) -> None:
        """Visualize the given pointcloud if the visualizer is active."""
        if self.uncalled_while_active is None:
            return  # Exit if the visualizer isn't active

        self.o3d_pcd.points = o3d.utility.Vector3dVector(pointcloud.points)

        if self.uncalled_while_active:
            self.vis.add_geometry(self.o3d_pcd)
            self.uncalled_while_active = False
        else:
            self.vis.update_geometry(self.o3d_pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
