from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .detection import detect_features, compute_optimal_scale, SlopeRegion
from .deformation import deform_mesh, apply_scale, DeformationResult


@dataclass
class SlopeConfig:
    planar_deg_err: float = 10.0
    normal_deg_err: float = 1.0
    min_area_fraction: float = 0.01


@dataclass
class SlopeResult:
    """Output of the slope detection + deformation pipeline."""
    mesh: o3d.geometry.TriangleMesh
    scale: float
    world_dim: tuple[int, int, int]
    assignments: list[tuple[SlopeRegion, list[dict]]]
    regions: list[SlopeRegion]
    deformation: DeformationResult | None
    region_n_steps: list[int] | None = None


def prepare_slopes(
    mesh: o3d.geometry.TriangleMesh,
    resolution: int = 20,
    cfg: SlopeConfig = SlopeConfig(),
) -> SlopeResult:
    features = detect_features(
        mesh,
        planar_deg_err=cfg.planar_deg_err,
        normal_deg_err=cfg.normal_deg_err,
        min_area_fraction=cfg.min_area_fraction,
    )

    optimal_scale, assignments = compute_optimal_scale(
        features.regions, default_scale=resolution,
    )
    s = int(optimal_scale)

    deformation = None
    if assignments:
        deformation = deform_mesh(mesh, scale=optimal_scale, assignments=assignments, flat_planes=features.planes)
        triangles = np.asarray(mesh.triangles)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(deformation.deformed_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
    else:
        mesh = apply_scale(mesh, optimal_scale)

    # Z-scale for plate height compensation
    vertices = np.asarray(mesh.vertices)
    vertices[:, 2] *= 3.0
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Compute world_dim from actual mesh bounds (deformation can expand the mesh)
    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    world_dim = (
        max(s, int(np.ceil(extent[0]))),
        max(s, int(np.ceil(extent[1]))),
        max(s * 3, int(np.ceil(extent[2]))),
    )

    return SlopeResult(
        mesh=mesh,
        scale=optimal_scale,
        world_dim=world_dim,
        assignments=assignments,
        regions=features.regions,
        deformation=deformation,
        region_n_steps=None,
    )
