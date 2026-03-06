"""Define a class to rasterize 3D collision models into 2D occupancy masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError
from skimage.draw import polygon

from robotics_utils.collision_models.primitive_shapes import PrimitiveShape, primitive_to_mesh
from robotics_utils.geometry import Point2D
from robotics_utils.spatial import Pose3D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.motion_planning import DiscreteGrid2D
    from robotics_utils.states import ObjectKinematicState


class CollisionModelRasterizer:
    """Rasterize 3D collision models into 2D occupancy masks.

    This class converts 3D collision geometries (meshes and primitives) into
    2D Boolean masks for use in occupancy grids, enabling "object removal"
    for hypothetical world queries.
    """

    def __init__(self, min_height_m: float = 0.1, max_height_m: float = 2.0) -> None:
        """Initialize the rasterizer with height limits for rasterized objects.

        :param min_height_m: Minimum height (meters) of included object geometry (default: 0.1 m)
        :param max_height_m: Maximum height (meters) of included object geometry (default: 2.0 m)
        """
        self.min_height_m = min_height_m
        self.max_height_m = max_height_m

    def rasterize_object(
        self,
        obj: ObjectKinematicState,
        grid: DiscreteGrid2D,
    ) -> NDArray[np.bool_]:
        """Rasterize an object's collision model into a mask for an occupancy grid.

        :param obj: Kinematic state of the object to be rasterized
        :param grid: Structure of the occupancy grid for the rasterization
        :return: Boolean mask where True indicates the object footprint
        """
        if obj.pose.ref_frame != grid.origin.ref_frame:
            raise ValueError(
                f"Cannot rasterize object '{obj.name}' with pose in frame '{obj.pose.ref_frame}' "
                f"for an occupancy grid with an origin in frame '{grid.origin.ref_frame}'.",
            )

        mask = np.zeros((grid.height_cells, grid.width_cells), dtype=bool)

        for mesh in obj.collision_model.meshes:
            mesh_mask = self.rasterize_mesh(mesh, obj.pose, grid)
            mask |= mesh_mask

        for primitive, pose_o_p in zip(
            obj.collision_model.primitives,
            obj.collision_model.primitive_poses,
        ):
            transform_w_p = obj.pose.to_homogeneous_matrix() @ pose_o_p.to_homogeneous_matrix()
            pose_w_p = Pose3D.from_homogeneous_matrix(
                transform_w_p,
                ref_frame=obj.pose.ref_frame,
            )
            primitive_mask = self.rasterize_primitive(primitive, pose_w_p, grid)
            mask |= primitive_mask

        return mask

    def rasterize_mesh(
        self,
        mesh: trimesh.Trimesh,
        obj_pose: Pose3D,
        grid: DiscreteGrid2D,
    ) -> NDArray[np.bool_]:
        """Rasterize a collision mesh into a mask for an occupancy grid.

        Computes the convex hull of all mesh vertices whose z-extent overlaps
        the height range, then fills that polygon on the grid. This provides a
        conservative (never underestimates) approximation of the mesh's 2D footprint.

        :param mesh: Collision mesh to be rasterized
        :param obj_pose: Object pose in the world frame
        :param grid: Structure of the occupancy grid for the rasterization
        :return: Boolean mask where True indicates the mesh's footprint
        """
        mask = np.zeros((grid.height_cells, grid.width_cells), dtype=np.bool_)

        # Transform mesh vertices into the world frame
        transform_w_o = obj_pose.to_homogeneous_matrix()
        vertices_world = trimesh.transform_points(mesh.vertices, transform_w_o)

        # Check if the mesh's z-extent overlaps with the height range
        z_coords = vertices_world[:, 2]

        if z_coords.max() < self.min_height_m or z_coords.min() > self.max_height_m:
            return mask

        # Use all vertices for the 2D footprint because the mesh overlaps the height range
        vertices_2d = vertices_world[:, :2]

        for vertex in vertices_2d:
            grid_cell = grid.world_to_cell(Point2D.from_array(vertex))
            if grid.is_valid_cell(grid_cell):
                mask[grid_cell.row, grid_cell.col] = True

        if vertices_2d.shape[0] < 3:  # Not enough vertices to form a polygon; exit
            return mask

        try:
            hull = ConvexHull(vertices_2d)
        except QhullError:  # Degenerate case (e.g., collinear points); exit
            return mask

        occupied_grid_rows = []
        occupied_grid_cols = []
        hull_vertices = hull.points[hull.vertices]  # Get only the boundary vertices, in order

        # Sample points along each hull edge to ensure that all cells along the edge are marked
        for i, start_point in enumerate(hull_vertices):
            end_point = hull_vertices[(i + 1) % len(hull_vertices)]
            edge_vec = end_point - start_point
            edge_length_m = np.linalg.norm(edge_vec)

            # Sample at intervals smaller than grid resolution to guarantee cell coverage
            num_samples = max(2, int(np.ceil(edge_length_m / grid.resolution_m)) + 1)
            for t in np.linspace(0.0, 1.0, num_samples):
                sample_point = start_point + t * edge_vec
                cell = grid.world_to_cell(Point2D.from_array(sample_point))
                occupied_grid_rows.append(cell.row)
                occupied_grid_cols.append(cell.col)
                if grid.is_valid_cell(cell):
                    mask[cell.row, cell.col] = True

        rr, cc = polygon(r=occupied_grid_rows, c=occupied_grid_cols, shape=mask.shape)
        mask[rr, cc] = True

        return mask

    def rasterize_primitive(
        self,
        primitive: PrimitiveShape,
        primitive_pose: Pose3D,
        grid: DiscreteGrid2D,
    ) -> NDArray[np.bool_]:
        """Rasterize a primitive collision geometry into a mask for an occupancy grid.

        :param primitive: Primitive shape to be rasterized
        :param primitive_pose: Primitive pose in the world frame
        :param grid: Structure of the occupancy grid for the rasterization
        :return: Boolean mask where True indicates the primitive's footprint
        """
        mesh = primitive_to_mesh(primitive)
        return self.rasterize_mesh(mesh, primitive_pose, grid)
