"""Define a class to rasterize 3D collision models into 2D occupancy grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError
from skimage.draw import polygon

from robotics_utils.collision_models import Box, Cylinder, PrimitiveShape, Sphere
from robotics_utils.geometry import Point2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.motion_planning import DiscreteGrid2D
    from robotics_utils.spatial import Pose3D
    from robotics_utils.states import ObjectKinematicState


class CollisionModelRasterizer:
    """Rasterize 3D collision models into 2D occupancy grid masks.

    This class converts 3D collision geometries (meshes and primitives) into
    2D Boolean masks for use in occupancy grids, enabling "object removal"
    for hypothetical world queries.
    """

    @staticmethod
    def rasterize_object(
        obj: ObjectKinematicState,
        grid: DiscreteGrid2D,
        min_height_m: float,
        max_height_m: float,
    ) -> NDArray[np.bool]:
        """Rasterize an object's collision model into a mask for an occupancy grid.

        :param obj: Kinematic state of the object to be rasterized
        :param grid: Structure of the occupancy grid for the rasterization
        :param min_height_m: Minimum height (meters) of the object to include
        :param max_height_m: Maximum height (meters) of the object to include
        :return: Boolean mask where True indicates the object footprint
        """
        if obj.pose.ref_frame != grid.origin.ref_frame:
            raise ValueError(
                f"Cannot rasterize object '{obj.name}' with pose in frame '{obj.pose.ref_frame}' "
                f"for an occupancy grid with an origin in frame '{grid.origin.ref_frame}'.",
            )

        mask = np.zeros((grid.height_cells, grid.width_cells), dtype=bool)

        for mesh in obj.collision_model.meshes:
            mesh_mask = CollisionModelRasterizer.rasterize_mesh(
                mesh=mesh,
                obj_pose=obj.pose,
                grid=grid,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            )
            mask |= mesh_mask

        for primitive in obj.collision_model.primitives:
            primitive_mask = CollisionModelRasterizer.rasterize_primitive(
                primitive=primitive,
                obj_pose=obj.pose,
                grid=grid,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            )
            mask |= primitive_mask

        return mask

    @staticmethod
    def rasterize_mesh(
        mesh: trimesh.Trimesh,
        obj_pose: Pose3D,
        grid: DiscreteGrid2D,
        min_height_m: float,
        max_height_m: float,
    ) -> NDArray[np.bool]:
        """Rasterize a collision mesh into a mask for an occupancy grid.

        Computes the convex hull of all mesh vertices within the height range,
        then fills that polygon on the grid. This provides a conservative
        (never underestimates) approximation of the mesh's 2D footprint.

        :param mesh: Collision mesh to be rasterized
        :param obj_pose: Object pose in the world frame
        :param grid: Structure of the occupancy grid for the rasterization
        :param min_height_m: Minimum height (meters) of the mesh to include
        :param max_height_m: Maximum height (meters) of the mesh to include
        :return: Boolean mask where True indicates the mesh's footprint
        """
        mask = np.zeros((grid.height_cells, grid.width_cells), dtype=np.bool)

        # Transform mesh vertices into the world frame
        transform_w_o = obj_pose.to_homogeneous_matrix()
        vertices_world = trimesh.transform_points(mesh.vertices, transform_w_o)

        # Filter vertices by height (z-coordinate in the world frame)
        z_coords = vertices_world[:, 2]
        height_mask = (z_coords >= min_height_m) & (z_coords <= max_height_m)
        filtered_vertices = vertices_world[height_mask]

        if filtered_vertices.shape[0] < 3:  # Not enough vertices to form a polygon
            for vertex in filtered_vertices:
                grid_cell = grid.world_to_cell(Point2D.from_array(vertex))
                if grid.is_valid_cell(grid_cell):
                    mask[grid_cell.row, grid_cell.col] = True
            return mask

        vertices_2d = filtered_vertices[:, :2]  # Project to 2D (x, y) coordinates

        try:
            hull = ConvexHull(vertices_2d)
        except QhullError:  # Degenerate case (e.g., collinear points); mark individual points
            for vertex in vertices_2d:
                grid_cell = grid.world_to_cell(Point2D.from_array(vertex))
                if grid.is_valid_cell(grid_cell):
                    mask[grid_cell.row, grid_cell.col] = True
            return mask

        occupied_grid_rows = []
        occupied_grid_cols = []
        hull_vertices = hull.points[hull.vertices]  # Get only the boundary vertices, in order
        for point in hull_vertices:
            row, col = grid.world_to_cell(Point2D.from_array(point))
            occupied_grid_rows.append(row)
            occupied_grid_cols.append(col)

        rr, cc = polygon(r=occupied_grid_rows, c=occupied_grid_cols, shape=mask.shape)
        mask[rr, cc] = True

        return mask

    @staticmethod
    def rasterize_primitive(
        primitive: PrimitiveShape,
        obj_pose: Pose3D,
        grid: DiscreteGrid2D,
        min_height_m: float,
        max_height_m: float,
    ) -> NDArray[np.bool]:
        """Rasterize a primitive collision geometry into a mask for an occupancy grid.

        :param primitive: Primitive shape to be rasterized
        :param obj_pose: Object pose in the world frame
        :param grid: Structure of the occupancy grid for the rasterization
        :param min_height_m: Minimum height (meters) of the primitive to include
        :param max_height_m: Maximum height (meters) of the primitive to include
        :return: Boolean mask where True indicates the primitive's footprint
        """
        mesh = CollisionModelRasterizer._primitive_to_mesh(primitive)
        return CollisionModelRasterizer.rasterize_mesh(
            mesh,
            obj_pose,
            grid,
            min_height_m,
            max_height_m,
        )

    @staticmethod
    def _primitive_to_mesh(primitive: PrimitiveShape) -> trimesh.Trimesh:
        """Convert a primitive shape to an equivalent trimesh mesh.

        :param primitive: Primitive shape to convert
        :return: Trimesh mesh representation of the primitive
        :raises ValueError: If primitive type is not supported
        """
        if isinstance(primitive, Box):
            return trimesh.primitives.Box(
                extents=[primitive.x_m, primitive.y_m, primitive.z_m],
            ).to_mesh()

        if isinstance(primitive, Sphere):
            return trimesh.primitives.Sphere(radius=primitive.radius_m).to_mesh()

        if isinstance(primitive, Cylinder):
            return trimesh.primitives.Cylinder(
                radius=primitive.radius_m,
                height=primitive.height_m,
            ).to_mesh()

        raise ValueError(f"Unexpected primitive shape type: {primitive} (type {type(primitive)})")
