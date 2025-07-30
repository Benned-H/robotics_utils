"""Utility classes and functions to support visualizations."""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from robotics_utils.kinematics.collision_models import CollisionModel


@dataclass(frozen=True)
class RGBA:
    """An RGB color with an alpha (A) value specifying transparency."""

    red: int  # Red value (between 0 and 255)
    green: int  # Green value (between 0 and 255)
    blue: int  # Blue value (between 0 and 255)
    alpha: int  # Alpha value (between 0 and 255)

    def __post_init__(self) -> None:
        """Verify that the constructed RGBA is valid."""
        for attr_name in ["red", "green", "blue", "alpha"]:
            attr_value = getattr(self, attr_name)
            if attr_value < 0 or attr_value > 255:
                raise ValueError(f"RGBA expects {attr_name} within [0, 255], got {attr_value}")


def create_axes_markers(length: float) -> trimesh.Trimesh:
    """Create RGB axes markers where the x-axis is red, y-axis is blue, and z-axis is green."""
    radius = length * 0.03  # Each axis will be a cylinder

    x_axis = trimesh.creation.cylinder(radius=radius, height=length)
    x_axis.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    x_axis.apply_translation([length / 2, 0, 0])
    x_axis.visual.face_colors = (255, 0, 0, 255)  # x-axis is red

    y_axis = trimesh.creation.cylinder(radius=radius, height=length)
    y_axis.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    y_axis.apply_translation([0, length / 2, 0])
    y_axis.visual.face_colors = (0, 255, 0, 255)  # y-axis is green

    z_axis = trimesh.creation.cylinder(radius=radius, height=length)
    z_axis.apply_translation([0, 0, length / 2])
    z_axis.visual.face_colors = (0, 0, 255, 255)  # z-axis is blue

    return trimesh.util.concatenate([x_axis, y_axis, z_axis])


def create_centroid_marker(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Create a marker to visualize the centroid of the given mesh."""
    marker = trimesh.creation.icosphere(radius=0.02, subdivisions=2)
    marker.apply_translation(mesh.centroid)
    marker.visual.face_colors = (255, 255, 0, 255)  # Yellow
    return marker


def create_text_label(
    text: str,
    position: np.NDArray,
    height: float = 0.05,
) -> trimesh.Trimesh:
    """Create a 3D text label at the specified position."""
    text_entity = trimesh.path.entities.Text(origin=position, text=text, height=height)
    path_2d = trimesh.path.Path2D(entities=[text_entity])
    text_extrusion = path_2d.extrude(0.01)
    text_extrusion.visual.face_colors = (0, 0, 0, 255)  # Black
    return text_extrusion


@dataclass(frozen=True)
class CollisionModelVizConfig:
    """Configuration for visualizing a collision model using trimesh."""

    # Select what to display
    show_meshes: bool = True
    show_primitives: bool = True
    show_origin: bool = True
    show_bounding_box: bool = True
    show_centroids: bool = True
    show_labels: bool = True

    # Visual properties
    mesh_color: RGBA = RGBA(100, 100, 100, 150)
    primitive_color: RGBA = RGBA(34, 219, 94, 100)
    bbox_color: RGBA = RGBA(242, 135, 37, 50)
    axes_length: float = 0.2

    # Scene properties
    resolution: tuple[int, int] = (1920, 1080)


def visualize_collision_model(
    model: CollisionModel,
    config: CollisionModelVizConfig | None = None,
) -> None:
    """Visualize the collision model based on the given configuration."""
    if config is None:
        config = CollisionModelVizConfig()

    scene = trimesh.Scene()
    labels: list[tuple[str, NDArray]] = []  # Track labels for the added geometry

    if config.show_meshes:
        for i, mesh_data in enumerate(model.meshes):
            mesh = mesh_data.mesh.copy()
            mesh.visual.face_colors = astuple(config.mesh_color)
            scene.add_geometry(mesh, node_name=f"mesh_{i}")
            if config.show_labels:
                labels.append((f"mesh_{i}", mesh.bounds.mean(axis=0)))

    if config.show_primitives:
        for i, primitive in enumerate(model.primitives):
            primitive_mesh = primitive.to_mesh()
            primitive_mesh.visual.face_colors = astuple(config.primitive_color)
            scene.add_geometry(primitive_mesh, node_name=f"primitive_{i}")
            if config.show_labels:
                labels.append((f"primitive_{i}", primitive_mesh.bounds.mean(axis=0)))

    if config.show_origin:
        axes = create_axes_markers(config.axes_length)
        scene.add_geometry(axes, node_name="origin_axes")
        if config.show_labels:
            labels.append(("origin_axes", np.array([0, 0, 0])))

    if config.show_bounding_box:
        bbox_mesh = model.aabb.to_mesh()
        bbox_mesh.visual.face_colors = astuple(config.bbox_color)
        scene.add_geometry(bbox_mesh, node_name="bounding_box")

    if config.show_centroids:
        for i, mesh_data in enumerate(model.meshes):
            centroid_marker = create_centroid_marker(mesh_data.mesh)
            scene.add_geometry(centroid_marker, node_name=f"centroid_{i}")
            if config.show_labels:
                labels.append((f"centroid_{i}", centroid_marker.bounds.mean(axis=0)))

    if config.show_labels and labels:
        for name, position in labels:
            label_pos = position + np.array([0, 0, 0.1])  # Place label slightly above the component
            label_mesh = create_text_label(text=name, position=label_pos)
            scene.add_geometry(label_mesh, node_name=f"label_{name}")

    scene.set_camera(resolution=config.resolution)
    scene.show()
