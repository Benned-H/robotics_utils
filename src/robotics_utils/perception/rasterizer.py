"""Define classes to rasterize 3D collision models into 2D occupancy grids."""

from __future__ import annotations

import numpy as np
import trimesh
from robotics_utils.collision_models import Box, Cylinder, PrimitiveShape, Sphere
from robotics_utils.geometry import Point2D
from robotics_utils.spatial import Pose2D, Pose3D
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
        resolution_m: float,
        origin: Pose2D,
        grid_shape: tuple[int, int],
        min_height_m: float,
        max_height_m: float,
    ) -> np.ndarray:
        """Rasterize object's collision model to Boolean mask.

        :param obj: Object to rasterize
        :param resolution_m: Grid resolution in meters
        :param origin: Grid origin pose
        :param grid_shape: Grid shape (height, width) in cells
        :param min_height_m: Minimum height to include
        :param max_height_m: Maximum height to include
        :return: Boolean mask where True indicates object footprint
        """
        height_cells, width_cells = grid_shape
        mask = np.zeros((height_cells, width_cells), dtype=bool)

        # Rasterize each mesh in the collision model
        for mesh in obj.collision_model.meshes:
            mesh_mask = CollisionModelRasterizer._rasterize_mesh_2d(
                mesh=mesh,
                obj_pose=obj.pose,
                resolution_m=resolution_m,
                origin=origin,
                grid_shape=grid_shape,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            )
            mask |= mesh_mask

        # Rasterize each primitive in the collision model
        for primitive in obj.collision_model.primitives:
            primitive_mask = CollisionModelRasterizer._rasterize_primitive_2d(
                primitive=primitive,
                obj_pose=obj.pose,
                resolution_m=resolution_m,
                origin=origin,
                grid_shape=grid_shape,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            )
            mask |= primitive_mask

        return mask

    @staticmethod
    def _rasterize_mesh_2d(
        mesh: trimesh.Trimesh,
        obj_pose: Pose3D,
        resolution_m: float,
        origin: Pose2D,
        grid_shape: tuple[int, int],
        min_height_m: float,
        max_height_m: float,
    ) -> np.ndarray:
        """Rasterize mesh into 2D grid by projecting vertices.

        :param mesh: Trimesh to rasterize
        :param obj_pose: Object pose in world frame
        :param resolution_m: Grid resolution
        :param origin: Grid origin
        :param grid_shape: Grid shape (height, width)
        :param min_height_m: Minimum height
        :param max_height_m: Maximum height
        :return: Boolean mask
        """
        height_cells, width_cells = grid_shape
        mask = np.zeros((height_cells, width_cells), dtype=bool)

        # Transform mesh vertices to world frame
        transform_matrix = obj_pose.to_homogeneous_matrix()
        vertices_world = trimesh.transform_points(mesh.vertices, transform_matrix)

        # Filter vertices by height (z coordinate in world frame)
        z_coords = vertices_world[:, 2]
        height_mask = (z_coords >= min_height_m) & (z_coords <= max_height_m)
        filtered_vertices = vertices_world[height_mask]

        if filtered_vertices.shape[0] == 0:
            return mask  # No vertices in height range

        # Project to 2D (x, y)
        vertices_2d = filtered_vertices[:, :2]

        # Convert to grid coordinates
        grid_coords = []
        for vertex in vertices_2d:
            point = Point2D(vertex[0], vertex[1])
            grid_x, grid_y = CollisionModelRasterizer._world_to_grid(
                point, origin, resolution_m
            )
            if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                grid_coords.append((grid_x, grid_y))

        if len(grid_coords) == 0:
            return mask  # No vertices in grid bounds

        # Get bounding box and fill it conservatively
        # For a more accurate approach, we would compute the 2D convex hull
        # and rasterize the polygon, but for simplicity we'll use the bounding box
        grid_xs = [coord[0] for coord in grid_coords]
        grid_ys = [coord[1] for coord in grid_coords]

        min_x = max(0, min(grid_xs))
        max_x = min(width_cells - 1, max(grid_xs))
        min_y = max(0, min(grid_ys))
        max_y = min(height_cells - 1, max(grid_ys))

        # Fill bounding box
        mask[min_y : max_y + 1, min_x : max_x + 1] = True

        return mask

    @staticmethod
    def _rasterize_primitive_2d(
        primitive: PrimitiveShape,
        obj_pose: Pose3D,
        resolution_m: float,
        origin: Pose2D,
        grid_shape: tuple[int, int],
        min_height_m: float,
        max_height_m: float,
    ) -> np.ndarray:
        """Rasterize primitive shape into 2D grid.

        :param primitive: Primitive shape to rasterize
        :param obj_pose: Object pose in world frame
        :param resolution_m: Grid resolution
        :param origin: Grid origin
        :param grid_shape: Grid shape (height, width)
        :param min_height_m: Minimum height
        :param max_height_m: Maximum height
        :return: Boolean mask
        """
        if isinstance(primitive, Box):
            return CollisionModelRasterizer._rasterize_box_2d(
                primitive, obj_pose, resolution_m, origin, grid_shape, min_height_m, max_height_m
            )
        if isinstance(primitive, Sphere):
            return CollisionModelRasterizer._rasterize_sphere_2d(
                primitive, obj_pose, resolution_m, origin, grid_shape, min_height_m, max_height_m
            )
        if isinstance(primitive, Cylinder):
            return CollisionModelRasterizer._rasterize_cylinder_2d(
                primitive, obj_pose, resolution_m, origin, grid_shape, min_height_m, max_height_m
            )

        # Unknown primitive type - return empty mask
        height_cells, width_cells = grid_shape
        return np.zeros((height_cells, width_cells), dtype=bool)

    @staticmethod
    def _rasterize_box_2d(
        box: Box,
        obj_pose: Pose3D,
        resolution_m: float,
        origin: Pose2D,
        grid_shape: tuple[int, int],
        min_height_m: float,
        max_height_m: float,
    ) -> np.ndarray:
        """Rasterize box primitive."""
        height_cells, width_cells = grid_shape
        mask = np.zeros((height_cells, width_cells), dtype=bool)

        # Check if box intersects height range
        box_min_z = obj_pose.z - box.z_m / 2.0
        box_max_z = obj_pose.z + box.z_m / 2.0

        if box_max_z < min_height_m or box_min_z > max_height_m:
            return mask  # Box outside height range

        # Get box corners in object frame
        half_x = box.x_m / 2.0
        half_y = box.y_m / 2.0

        corners_obj = np.array([
            [half_x, half_y, 0],
            [half_x, -half_y, 0],
            [-half_x, -half_y, 0],
            [-half_x, half_y, 0],
        ])

        # Transform to world frame
        transform_matrix = obj_pose.to_homogeneous_matrix()
        corners_world = trimesh.transform_points(corners_obj, transform_matrix)

        # Project to 2D and convert to grid coordinates
        grid_corners = []
        for corner in corners_world:
            point = Point2D(corner[0], corner[1])
            grid_x, grid_y = CollisionModelRasterizer._world_to_grid(
                point, origin, resolution_m
            )
            grid_corners.append((grid_x, grid_y))

        # Get bounding box
        grid_xs = [c[0] for c in grid_corners]
        grid_ys = [c[1] for c in grid_corners]

        min_x = max(0, int(np.floor(min(grid_xs))))
        max_x = min(width_cells - 1, int(np.ceil(max(grid_xs))))
        min_y = max(0, int(np.floor(min(grid_ys))))
        max_y = min(height_cells - 1, int(np.ceil(max(grid_ys))))

        # Fill bounding box
        mask[min_y : max_y + 1, min_x : max_x + 1] = True

        return mask

    @staticmethod
    def _rasterize_sphere_2d(
        sphere: Sphere,
        obj_pose: Pose3D,
        resolution_m: float,
        origin: Pose2D,
        grid_shape: tuple[int, int],
        min_height_m: float,
        max_height_m: float,
    ) -> np.ndarray:
        """Rasterize sphere primitive."""
        height_cells, width_cells = grid_shape
        mask = np.zeros((height_cells, width_cells), dtype=bool)

        # Check if sphere intersects height range
        sphere_min_z = obj_pose.z - sphere.radius_m
        sphere_max_z = obj_pose.z + sphere.radius_m

        if sphere_max_z < min_height_m or sphere_min_z > max_height_m:
            return mask  # Sphere outside height range

        # Project sphere center to 2D grid
        center_2d = Point2D(obj_pose.x, obj_pose.y)
        center_grid_x, center_grid_y = CollisionModelRasterizer._world_to_grid(
            center_2d, origin, resolution_m
        )

        # Compute radius in grid cells
        radius_cells = int(np.ceil(sphere.radius_m / resolution_m))

        # Fill circle
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx**2 + dy**2 <= radius_cells**2:
                    grid_x = center_grid_x + dx
                    grid_y = center_grid_y + dy
                    if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                        mask[grid_y, grid_x] = True

        return mask

    @staticmethod
    def _rasterize_cylinder_2d(
        cylinder: Cylinder,
        obj_pose: Pose3D,
        resolution_m: float,
        origin: Pose2D,
        grid_shape: tuple[int, int],
        min_height_m: float,
        max_height_m: float,
    ) -> np.ndarray:
        """Rasterize cylinder primitive (same as sphere for 2D projection)."""
        height_cells, width_cells = grid_shape
        mask = np.zeros((height_cells, width_cells), dtype=bool)

        # Check if cylinder intersects height range
        cylinder_min_z = obj_pose.z - cylinder.height_m / 2.0
        cylinder_max_z = obj_pose.z + cylinder.height_m / 2.0

        if cylinder_max_z < min_height_m or cylinder_min_z > max_height_m:
            return mask  # Cylinder outside height range

        # Project cylinder to circle (same as sphere)
        center_2d = Point2D(obj_pose.x, obj_pose.y)
        center_grid_x, center_grid_y = CollisionModelRasterizer._world_to_grid(
            center_2d, origin, resolution_m
        )

        # Compute radius in grid cells
        radius_cells = int(np.ceil(cylinder.radius_m / resolution_m))

        # Fill circle
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx**2 + dy**2 <= radius_cells**2:
                    grid_x = center_grid_x + dx
                    grid_y = center_grid_y + dy
                    if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                        mask[grid_y, grid_x] = True

        return mask

    @staticmethod
    def _world_to_grid(point: Point2D, origin: Pose2D, resolution_m: float) -> tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        # Transform point relative to grid origin
        dx = point.x - origin.x
        dy = point.y - origin.y

        # Rotate by -origin.yaw to align with grid axes
        cos_yaw = np.cos(-origin.yaw_rad)
        sin_yaw = np.sin(-origin.yaw_rad)

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        # Convert to grid indices
        grid_x = int(np.floor(local_x / resolution_m))
        grid_y = int(np.floor(local_y / resolution_m))

        return grid_x, grid_y
