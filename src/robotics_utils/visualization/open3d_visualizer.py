"""Define a class to visualize pointclouds and other 3D geometries using Open3D."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

from robotics_utils.reconstruction import PlaneEstimate, Pointcloud

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from robotics_utils.geometry import Plane3D


class Open3DVisualizer:
    """A manager context for live visualization of pointclouds and 3D geometries using Open3D."""

    def __init__(self, window_name: str = "Open3D Visualization") -> None:
        """Initialize an Open3D visualizer for pointclouds and other geometries.

        :param window_name: Name for the visualization window
        """
        self.vis = o3d.visualization.Visualizer()
        self.window_name = window_name

        self.geometries: dict[str, o3d.geometry.Geometry] = {}
        """A map from geometry names to the corresponding Open3D geometries."""

        self.active: bool = False

    def __enter__(self) -> Self:
        """Enter a managed context for live pointcloud visualization."""
        self.vis.create_window(window_name=self.window_name)
        self.active = True

        # Set Open3D render options
        options = self.vis.get_render_option()
        options.mesh_show_wireframe = True
        options.mesh_show_back_face = True

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
        self.active = False
        self.geometries.clear()
        return None

    def add_geometry(
        self,
        name: str,
        geometry: o3d.geometry.Geometry,
        *,
        update_display: bool = True,
    ) -> None:
        """Add or update a named geometry in the visualization (if it's active).

        :param name: Unique identifier for the geometry
        :param geometry: Open3D geometry to add or update
        :param update_display: Whether to update the display after adding (defaults to True)
        """
        if not self.active:
            return

        if name in self.geometries:
            self.vis.remove_geometry(self.geometries[name], reset_bounding_box=False)

        self.geometries[name] = geometry
        self.vis.add_geometry(geometry, reset_bounding_box=(len(self.geometries) == 1))

        if update_display:
            self._update_display()

    def remove_geometry(self, name: str, *, update_display: bool = True) -> None:
        """Remove an added geometry by name.

        :param name: Identifier for the geometry to be removed
        :param update_display: Whether to update the display after removing (defaults to True)
        """
        if not self.active or name not in self.geometries:
            return

        self.vis.remove_geometry(self.geometries[name], reset_bounding_box=False)
        del self.geometries[name]
        if update_display:
            self._update_display()

    def add_pointcloud(
        self,
        name: str,
        pointcloud: Pointcloud,
        *,
        update_display: bool = True,
    ) -> None:
        """Add the given pointcloud to the visualization.

        :param name: Unique identifier for the pointcloud
        :param pointcloud: Pointcloud to visualize
        :param update_display: Whether to update the display after adding (defaults to True)
        """
        o3d_pcd = pointcloud.to_o3d()

        if name not in self.geometries:  # First time adding this name
            self.add_geometry(name, geometry=o3d_pcd, update_display=update_display)
            return

        # Otherwise, check if the existing geometry is a PointCloud
        if isinstance(self.geometries[name], o3d.geometry.PointCloud):
            self.geometries[name].points = o3d_pcd.points
            self.geometries[name].colors = o3d_pcd.colors

            self.vis.update_geometry(self.geometries[name])
            if update_display:
                self._update_display()
        else:  # Name exists but it's not a PointCloud
            self.remove_geometry(name, update_display=False)
            self.add_geometry(name, geometry=o3d_pcd, update_display=update_display)

    def add_triangle_mesh(
        self,
        name: str,
        mesh: o3d.geometry.TriangleMesh,
        *,
        update_display: bool = True,
    ) -> None:
        """Add the given triangle mesh to the visualization.

        :param name: Unique identifier for the triangle mesh
        :param mesh: Triangle mesh to visualize
        :param update_display: Whether to update the display after adding (defaults to True)
        """
        if name not in self.geometries:  # First time adding this name
            self.add_geometry(name, geometry=mesh, update_display=update_display)
            return

        # Otherwise, check if the existing geometry is a TriangleMesh
        if isinstance(self.geometries[name], o3d.geometry.TriangleMesh):
            self.geometries[name].triangle_normals = mesh.triangle_normals
            self.geometries[name].triangles = mesh.triangles
            self.geometries[name].vertex_colors = mesh.vertex_colors
            self.geometries[name].vertex_normals = mesh.vertex_normals
            self.geometries[name].vertices = mesh.vertices

            self.vis.update_geometry(self.geometries[name])
            if update_display:
                self._update_display()
        else:  # Name exists but it's not a TriangleMesh
            self.remove_geometry(name, update_display=False)
            self.add_geometry(name, geometry=mesh, update_display=update_display)

    def add_plane(
        self,
        name: str,
        plane: Plane3D,
        size_m: float = 0.2,
        color: tuple[float, float, float] = (0.56, 0.0, 1.0),
        *,
        update_display: bool = True,
    ) -> None:
        """Add the given plane into the visualization.

        :param name: Unique identifier for the plane
        :param plane: Plane3D to be visualized
        :param size_m: Size of the plane's mesh visualization (side length in meters)
        :param color: RGB color tuple with values in the range [0, 1] (defaults to electric violet)
        :param update_display: Whether to update the display after adding (defaults to True)
        """
        plane_mesh = Open3DVisualizer.create_plane_mesh(plane=plane, size_m=size_m, color=color)
        self.add_triangle_mesh(name, mesh=plane_mesh, update_display=update_display)

    def add_coordinate_frame(
        self,
        name: str,
        origin: np.typing.NDArray[np.float64],
        size_m: float = 0.1,
        *,
        update_display: bool = True,
    ) -> None:
        """Add a coordinate frame to the visualization.

        :param name: Unique identifier for the coordinate frame
        :param origin: 3D position for the frame origin as an array of shape (3,)
        :param size_m: Size of the coordinate frame axes (in meters)
        :param update_display: Whether to update the display after adding (defaults to True)
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_m, origin=origin)
        self.add_triangle_mesh(name, mesh=frame, update_display=update_display)

    def add_plane_estimate(
        self,
        name: str,
        estimate: PlaneEstimate,
        plane_size_m: float = 0.2,
        axes_size_m: float = 0.1,
    ) -> None:
        """Visualize a plane estimated from a pointcloud with inlier highlighting.

        :param name: Unique identifier for the plane estimate
        :param estimate: PlaneEstimate containing pointcloud, estimated plane, and inlier indices
        :param plane_size_m: Size of the square used to visualize the plane (side length in meters)
        :param axes_size_m: Size (meters) of the coordinate axes in the visualization
        """
        vis_colors = np.zeros((len(estimate.pointcloud), 3), dtype=np.uint8)  # (N, 3)
        vis_colors[:] = [120, 120, 120]  # Gray for outliers

        # Use RGB values if the pointcloud has them for inliers, otherwise use green
        vis_colors[estimate.inlier_indices] = (
            [0, 255, 0]
            if estimate.pointcloud.colors is None
            else estimate.pointcloud.colors[estimate.inlier_indices]
        )

        vis_pointcloud = Pointcloud(estimate.pointcloud.points, vis_colors)
        self.add_pointcloud(f"{name}-pointcloud", vis_pointcloud, update_display=False)

        self.add_plane(f"{name}-plane", estimate.plane, plane_size_m, update_display=False)

        self.add_coordinate_frame(f"{name}-frame", origin=estimate.plane.point, size_m=axes_size_m)

    def remove_plane_estimate(self, name: str) -> None:
        """Remove all geometry associated with the named plane estimate from the visualization."""
        self.remove_geometry(f"{name}-pointcloud", update_display=False)
        self.remove_geometry(f"{name}-plane", update_display=False)
        self.remove_geometry(f"{name}-frame")

    def _update_display(self) -> None:
        """Update the visualization display."""
        self.vis.poll_events()
        self.vis.update_renderer()

    @staticmethod
    def create_plane_mesh(
        plane: Plane3D,
        size_m: float,
        color: tuple[float, float, float],
    ) -> o3d.geometry.TriangleMesh:
        """Create a rectangular mesh visualization of the given plane.

        :param plane: Plane to be visualized
        :param size_m: Size of the rectangular visualization (side length in meters)
        :param color: RGB color tuple with values in the range [0, 1]
        :return: Open3D mesh representing the plane
        """
        # 1. Find two orthogonal basis vectors in the plane
        basis = plane.find_basis()
        u = basis.u
        v = basis.v

        # Create corners of the rectangular mesh, then the mesh
        corner_coords = [(u + v), (-u + v), (-u - v), (u - v)]
        corners = [plane.point + (c * size_m / 2) for c in corner_coords]

        vertices = np.asarray(corners)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()

        return mesh
