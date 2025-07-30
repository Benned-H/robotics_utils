"""Define utility functions for loading meshes from file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import pyvista as pv
import trimesh
from fast_simplification import simplify_mesh


@dataclass(frozen=True)
class MeshImportConfig:
    """Configuration for a mesh import process."""

    max_faces: int = 50000
    """Maximum number of faces kept in the imported mesh."""

    max_simplification_iters: int = 5
    """Maximum number of mesh simplification loops to try."""

    aggression: int = 6
    """Aggression used during mesh decimation (between 0 and 10)."""


def load_mesh(mesh_path: Path, config: MeshImportConfig | None = None) -> trimesh.Trimesh:
    """Load a mesh from the given filepath.

    :param mesh_path: Filepath to a mesh file (e.g., obj)
    :param config: Configuration for the mesh import (optional; uses default values if None)
    :return: Loaded trimesh.Trimesh instance
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    if config is None:
        config = MeshImportConfig()

    suffix = mesh_path.suffix.strip(".").lower()
    if suffix in {"stp", "step"}:
        mesh_path = convert_step_to_stl(step_path=mesh_path)

    pv_data = pv.read(mesh_path, progress_bar=True)
    initial_face_count = pv_data.n_faces_strict

    curr_aggression = config.aggression
    for _ in range(config.max_simplification_iters):
        pre_face_count = pv_data.n_faces_strict
        if pre_face_count <= config.max_faces:
            break

        pv_data = simplify_mesh(pv_data, target_count=config.max_faces, agg=config.aggression)

        print(
            f"Mesh was simplified from {pre_face_count} faces to {pv_data.n_faces_strict} faces "
            f"(aggression {curr_aggression})",
        )

        curr_aggression = min(curr_aggression + 1, 10)  # Bump up aggression each iteration

    if pv_data.n_faces_strict > config.max_faces:
        pre_face_count = pv_data.n_faces_strict
        reduction = 1.0 - config.max_faces / pre_face_count
        pv_data = simplify_mesh(pv_data, target_reduction=reduction, agg=curr_aggression)

        print(
            f"Mesh was simplified from {pre_face_count} faces to {pv_data.n_faces_strict} faces "
            f"(aggression {curr_aggression})",
        )

    final_face_count = pv_data.n_faces_strict

    if final_face_count < initial_face_count:
        simplified_path = mesh_path.parent / f"{mesh_path.stem}_simplified_{final_face_count}.stl"
        pv_data.save(simplified_path)
        print(f"Simplified mesh was saved to file: {simplified_path}")

        mesh_path = simplified_path

    print(f"Loading mesh from file {mesh_path}...")
    mesh = trimesh.load_mesh(mesh_path)
    print(f"Loaded mesh with {len(mesh.faces)} from {mesh_path}")

    return mesh


def convert_step_to_stl(step_path: Path) -> Path:
    """Convert the given STEP file into an STL file.

    :param step_path: Path to an existing STEP file
    :return: Path to the created STL file
    """
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    suffix = step_path.suffix.strip(".").lower()
    if suffix not in {"stp", "step"}:
        raise RuntimeError(f"Cannot convert non-STEP file to STL: {step_path}")

    output_path = step_path.parent / f"{step_path.stem}.stl"
    if output_path.exists():
        return output_path  # Skip conversion if an STL file already exists

    print(f"Loading STEP file from path '{step_path}'...")
    step = cq.importers.importStep(str(step_path))
    print(f"Exporting STL file to path '{output_path}'...")
    cq.exporters.export(step, str(output_path))

    return output_path
