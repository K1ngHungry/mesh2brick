from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .detection import detect_features, compute_optimal_scale, SlopeRegion
from .deformation import deform_mesh, apply_scale, DeformationResult


@dataclass
class SlopeConfig:
    """Configuration for slope detection.

    planar_err: Max angle (degrees) between a face normal and the region's
        average normal for the face to be considered coplanar.
    normal_err: Max angle (degrees) between adjacent face normals for them
        to be grouped into the same region.
    min_area: Minimum region area as a fraction of total mesh area.
    """
    planar_err: float = 10.0
    normal_err: float = 1.0
    min_area: float = 0.01


@dataclass
class SlopeResult:
    """Output of the slope detection + deformation pipeline.

    mesh: The scaled (and optionally deformed) mesh ready for voxelization.
    scale: The voxel scale factor applied to the mesh
    world_dim: Voxel grid dimensions (x, y, z) derived from mesh bounds
    assignments: Each detected slope region paired with its matching brick
    regions: All detected slope regions
    deformation: Vertex deformation details, or None if no slopes found
    """
    mesh: o3d.geometry.TriangleMesh
    scale: float
    world_dim: tuple[int, int, int]
    assignments: list[tuple[SlopeRegion, list[dict]]]
    regions: list[SlopeRegion]
    deformation: DeformationResult | None


def prepare_slopes(
    mesh: o3d.geometry.TriangleMesh,
    resolution: int = 20,
    cfg: SlopeConfig = SlopeConfig(),
) -> SlopeResult:
    """Detect slope regions, scale and deform the mesh for brick tiling.

    1. Applies Z-scale (x3) for plate height compensation BEFORE detection.
    2. Detects slope regions and flat planes on the mesh.
    3. Computes an optimal voxel scale that aligns slope dimensions to
       available brick sizes.
    4. If slopes were found, deforms the mesh so slope surfaces align to
       the voxel grid; otherwise just applies uniform scaling.
    5. Computes world_dim from the resulting mesh bounds.
    """
    # Apply Z-scale (3x) for plate height compensation BEFORE detection
    vertices = np.asarray(mesh.vertices)
    vertices[:, 2] *= 3.0
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    features = detect_features(
        mesh,
        planar_err=cfg.planar_err,
        normal_err=cfg.normal_err,
        min_area=cfg.min_area,
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
    )
