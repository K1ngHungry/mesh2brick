"""Model Deformation (Grid Alignment).

Deforms a mesh so that detected slope regions snap to integer voxel grid
coordinates with dimensions that are exact multiples of slope brick sizes.
"""
import copy
import math
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.optimize import minimize

from mesh2brick.slope_detection import (
    SlopeRegion,
    match_slope_to_bricks,
)
from mesh2brick.mesh_utils import (
    build_face_adjacency,
    normal_angle_diff,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Plane:
    """A detected flat (axis-aligned) surface region."""
    face_indices: list[int]
    vertex_indices: list[int]
    normal: np.ndarray
    offset: float


@dataclass
class SplitVertex:
    """A vertex shared between a slope region and the rest of the mesh."""
    original_index: int
    region_index: int        # which slope region owns the slope-side copy
    slope_var_index: int
    mesh_var_index: int


@dataclass
class DeformationResult:
    """Output of the deformation pipeline."""
    deformed_vertices: np.ndarray
    split_vertices: list[SplitVertex]
    flat_planes: list[Plane]
    slope_corner_indices: list[int]
    final_energy: float


def apply_scale(
    mesh: o3d.geometry.TriangleMesh,
    scale: float,
) -> o3d.geometry.TriangleMesh:
    """Scale mesh uniformly by *scale*. Returns a new mesh (does not mutate input).

    After this, 1 unit in XY ≈ 1 stud.
    """
    new_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(new_mesh.vertices)
    vertices *= scale
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return new_mesh



# 6 axis directions to check for flat planes
_AXIS_NORMALS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=float)


def _to_axis(normal: np.ndarray, planar_deg_err: float) -> np.ndarray | None:
    """If *normal* is within *planar_deg_err* of an axis direction, return
    that axis direction; otherwise return None."""
    dots = _AXIS_NORMALS @ normal
    best_idx = int(np.argmax(dots))
    best_dot = dots[best_idx]
    angle = math.degrees(math.acos(np.clip(best_dot, -1.0, 1.0)))
    if angle <= planar_deg_err:
        return _AXIS_NORMALS[best_idx].copy()
    return None


def detect_planes(
    mesh: o3d.geometry.TriangleMesh,
    planar_deg_err: float = 10.0,
    normal_deg_err: float = 5.0,
    offset_tol: float = 0.5,
    min_faces: int = 3,
) -> list[Plane]:
    """Detect flat (axis-aligned) surface regions on the mesh.

    Finds connected groups of faces whose normals are near one 
    of the 6 axis directions and that lie on the same plane.

    Args:
        mesh: Scaled mesh (post apply_scale).
        planar_deg_err: Max angle from an axis to qualify as flat.
        normal_deg_err: Max angle diff for BFS grouping.
        offset_tol: Max difference in plane offset (n·centroid) for BFS.
        min_faces: Minimum faces to form a plane group.

    Returns:
        List of Plane objects.
    """
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    if len(faces) == 0:
        return []

    # Compute face centroids
    centroids = vertices[faces].mean(axis=1)  # (n_faces, 3)

    # Find faces that are near an axis direction
    flat_faces: dict[int, np.ndarray] = {}
    for fi in range(len(faces)):
        axis = _to_axis(normals[fi], planar_deg_err)
        if axis is not None:
            flat_faces[fi] = axis

    if not flat_faces:
        return []

    # Build adjacency restricted to flat faces
    adjacency = build_face_adjacency(faces, set(flat_faces.keys()))

    # BFS group: same axis normal + similar plane offset
    visited: set[int] = set()
    groups: list[tuple[list[int], np.ndarray]] = []  # (face_indices, axis_normal)

    for start_face in flat_faces:
        if start_face in visited:
            continue
        seed_axis = flat_faces[start_face]
        seed_offset = float(np.dot(seed_axis, centroids[start_face]))

        queue = [start_face]
        visited.add(start_face)
        group = []
        head = 0
        while head < len(queue):
            face = queue[head]
            head += 1
            group.append(face)
            for neighbor in adjacency.get(face, []):
                if neighbor in visited:
                    continue
                if neighbor not in flat_faces:
                    continue
                # Check same axis direction
                if normal_angle_diff(seed_axis, normals[neighbor]) > normal_deg_err:
                    continue
                # Check same plane offset
                nbr_offset = float(np.dot(seed_axis, centroids[neighbor]))
                if abs(nbr_offset - seed_offset) > offset_tol:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        if len(group) >= min_faces:
            groups.append((group, seed_axis))

    # Build Plane objects
    planes: list[Plane] = []
    for face_list, axis_normal in groups:
        vert_indices = np.unique(faces[face_list]).tolist()
        plane_verts = vertices[vert_indices]
        offset = float(np.dot(axis_normal, plane_verts.mean(axis=0)))
        planes.append(Plane(
            face_indices=face_list,
            vertex_indices=vert_indices,
            normal=axis_normal,
            offset=offset,
        ))

    return planes


def _classify_vertices(
    faces: np.ndarray,
    assignments: list[tuple[SlopeRegion, list[dict]]],
) -> tuple[list[np.ndarray], list[int], dict[int, int]]:
    """Partition vertices into slope regions and identify split vertices.

    A *split vertex* is one shared between slope-region faces and non-slope faces

    Args:
        faces: (F, 3) triangle index array.
        assignments: (region, matched_bricks) pairs.

    Returns:
        (region_vert_indices, split_indices, vert_to_region):
        - region_vert_indices: unique vertex indices per region.
        - split_indices: sorted vertex indices shared between slope/non-slope.
        - vert_to_region: maps each slope vertex to its first owning region.
    """
    slope_faces: set[int] = set()
    for region, _bricks in assignments:
        slope_faces.update(region.face_indices)

    region_vert_indices: list[np.ndarray] = []
    slope_verts: set[int] = set()
    for region, _bricks in assignments:
        region_faces = faces[region.face_indices]
        vert_indices = np.unique(region_faces)
        region_vert_indices.append(vert_indices)
        slope_verts.update(vert_indices.tolist())

    non_slope_faces = set(range(len(faces))) - slope_faces
    non_slope_verts: set[int] = set()
    for fi in non_slope_faces:
        non_slope_verts.update(faces[fi].tolist())

    split_indices = sorted(slope_verts & non_slope_verts)

    vert_to_region: dict[int, int] = {}
    for ri, vert_arr in enumerate(region_vert_indices):
        for vi in vert_arr:
            vi_int = int(vi)
            if vi_int not in vert_to_region:
                vert_to_region[vi_int] = ri

    return region_vert_indices, split_indices, vert_to_region


def _resize_region(
    vertices: np.ndarray,
    target: np.ndarray,
    region: SlopeRegion,
    bricks: list[dict],
    vert_indices: np.ndarray,
) -> None:
    """Scale one region's vertices in *target* to dimensions that are multiples of brick sizes.
    """
    region_verts = vertices[vert_indices]
    if len(region_verts) == 0:
        return

    min_b = region_verts.min(axis=0)
    max_b = region_verts.max(axis=0)
    region_dims = max_b - min_b
    centroid = (min_b + max_b) / 2.0

    brick = min(bricks, key=lambda b: b['length'] * b['width'])

    if region.slope_direction in (0, 2):  # X-aligned slope
        length_axis, width_axis = 0, 1
    else:  # Y-aligned slope
        length_axis, width_axis = 1, 0
    height_axis = 2

    scale_factors = np.ones(3)
    for axis, brick_dim_key in [
        (length_axis, 'length'),
        (width_axis, 'width'),
        (height_axis, 'height'),
    ]:
        if region_dims[axis] > 0:
            brick_dim = brick[brick_dim_key]
            target_dim = math.ceil(region_dims[axis] / brick_dim) * brick_dim
            scale_factors[axis] = target_dim / region_dims[axis]

    for vi in vert_indices:
        target[vi] = centroid + (vertices[vi] - centroid) * scale_factors


def resize_slope_regions(
    mesh: o3d.geometry.TriangleMesh,
    assignments: list[tuple[SlopeRegion, list[dict]]],
) -> tuple[np.ndarray, list[SplitVertex], list[np.ndarray]]:
    """Resize slope regions to exact brick-multiple dimensions.

    For each assigned region, scales its vertices so that the region's bounding box 
    becomes an integer multiple of the matched brick dimensions.

    Also identifies split vertices: vertices shared between slope region
    faces and non-slope faces.

    Args:
        mesh: Scaled mesh (post apply_scale).
        assignments: (region, matched_bricks) pairs.

    Returns:
        (target_positions, split_vertices, region_vert_indices):
        - target_positions: (N, 3) copy of mesh vertices with slope vertices moved.
        - split_vertices: SplitVertex objects with region_index set.
        - region_vert_indices: list of vertex index arrays, one per region.
    """
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    target = vertices.copy()

    region_vert_indices, split_indices, vert_to_region = _classify_vertices(
        faces, assignments,
    )

    for ri, (region, bricks) in enumerate(assignments):
        _resize_region(vertices, target, region, bricks, region_vert_indices[ri])

    split_vertices = [
        SplitVertex(
            original_index=idx,
            region_index=vert_to_region.get(idx, 0),
            slope_var_index=-1,
            mesh_var_index=-1,
        )
        for idx in split_indices
    ]

    return target, split_vertices, region_vert_indices

def _energy_and_gradient(
    x: np.ndarray,
    resized_positions: np.ndarray,
    n_regions: int,
    region_corner_indices: list[list[int]],
    split_vertices: list[SplitVertex],
    flat_planes: list[Plane],
    split_plane_membership: dict[int, list[tuple[np.ndarray, float]]],
    plane_vertex_indices: list[int],
    plane_vertex_membership: dict[int, list[tuple[np.ndarray, float]]],
    lambda_t: float,
    lambda_p: float,
) -> tuple[float, np.ndarray]:
    """Compute E_deform and its gradient w.r.t. rigid-body translations.

    Returns:
        (E_deform, grad) — grad has same shape as x.
    """
    n_split = len(split_vertices)
    n_plane = len(plane_vertex_indices)
    n_vars = 3 * n_regions + 3 * n_split + 3 * n_plane
    split_offset = 3 * n_regions
    plane_offset = split_offset + 3 * n_split

    grad = np.zeros(n_vars)
    E_deform = 0.0

    # E_int (Eq 9): snap region corner vertices to integer grid
    for ri in range(n_regions):
        p_ri = x[3 * ri: 3 * ri + 3]
        for vi in region_corner_indices[ri]:
            v_i = resized_positions[vi] + p_ri
            diff = v_i - np.round(v_i)
            E_deform += np.dot(diff, diff)
            grad[3 * ri: 3 * ri + 3] += 2.0 * diff

    # E_topology (Eq 8): pull split vertex pairs (v_slope, v_mesh) together
    for j, sv in enumerate(split_vertices):
        ri = sv.region_index
        p_ri = x[3 * ri: 3 * ri + 3]
        v_slope = resized_positions[sv.original_index] + p_ri
        v_mesh = x[split_offset + 3 * j: split_offset + 3 * j + 3]
        diff = v_slope - v_mesh
        E_deform += lambda_t * np.dot(diff, diff)
        grad[3 * ri: 3 * ri + 3] += lambda_t * 2.0 * diff
        grad[split_offset + 3 * j: split_offset + 3 * j + 3] += lambda_t * (-2.0 * diff)

    # E_planarity (Eq 10): keep mesh-side split copies on their flat planes
    for j, sv in enumerate(split_vertices):
        planes = split_plane_membership.get(sv.original_index, [])
        for n_i, v_0i in planes:
            v_mesh = x[split_offset + 3 * j: split_offset + 3 * j + 3]
            proj = float(np.dot(n_i, v_mesh)) - v_0i
            E_deform += lambda_p * proj * proj
            grad[split_offset + 3 * j: split_offset + 3 * j + 3] += (
                lambda_p * 2.0 * proj * n_i
            )

    # E_planarity for all plane vertices (non-slope vertices on detected planes)
    for k, vi in enumerate(plane_vertex_indices):
        v_pos = x[plane_offset + 3 * k: plane_offset + 3 * k + 3]
        # Planarity: keep on original plane(s)
        for n_i, v_0i in plane_vertex_membership[vi]:
            proj = float(np.dot(n_i, v_pos)) - v_0i
            E_deform += lambda_p * proj * proj
            grad[plane_offset + 3 * k: plane_offset + 3 * k + 3] += (
                lambda_p * 2.0 * proj * n_i
            )
        # Position regularization: stay near original position
        diff = v_pos - resized_positions[vi]
        E_deform += np.dot(diff, diff)
        grad[plane_offset + 3 * k: plane_offset + 3 * k + 3] += 2.0 * diff

    return E_deform, grad

def optimize_vertices(
    resized_positions: np.ndarray,
    region_vert_indices: list[np.ndarray],
    region_corner_indices: list[list[int]],
    split_vertices: list[SplitVertex],
    flat_planes: list[Plane],
    lambda_t: float = 5.0,
    lambda_p: float = 4.0,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Minimize E_deform using rigid-body translations per slope region.

    Planarity is enforced on ALL vertices belonging to detected flat planes,
    not just split vertices. This keeps entire walls/roofs coplanar during
    deformation.

    Args:
        resized_positions: (N, 3) vertex positions (slope vertices already resized).
        region_vert_indices: Vertex indices for each region.
        region_corner_indices: Corner vertex indices for each region.
        split_vertices: Split vertex pairs with region_index set.
        flat_planes: Detected flat planes.
        lambda_t: Weight for topology energy.
        lambda_p: Weight for planarity energy.
        max_iter: Max L-BFGS-B iterations.

    Returns:
        (optimized_positions, final_energy).
    """
    n_regions = len(region_vert_indices)
    n_split = len(split_vertices)

    if n_regions == 0:
        return resized_positions.copy(), 0.0

    # Collect all slope vertex indices
    slope_verts: set[int] = set()
    for vert_arr in region_vert_indices:
        slope_verts.update(int(v) for v in vert_arr)
    split_vert_set = {sv.original_index for sv in split_vertices}

    # Precompute plane membership for split vertices (mesh-side planarity)
    split_plane_membership: dict[int, list[tuple[np.ndarray, float]]] = {}
    for plane in flat_planes:
        vs = set(plane.vertex_indices)
        for sv in split_vertices:
            if sv.original_index in vs:
                if sv.original_index not in split_plane_membership:
                    split_plane_membership[sv.original_index] = []
                split_plane_membership[sv.original_index].append(
                    (plane.normal, plane.offset)
                )

    # Identify non-slope plane vertices: vertices on flat planes that are
    # NOT slope region vertices (split vertices are already handled above).
    plane_vertex_membership: dict[int, list[tuple[np.ndarray, float]]] = {}
    for plane in flat_planes:
        for vi in plane.vertex_indices:
            if vi in slope_verts:
                continue  # slope verts move rigidly, split verts already free
            if vi not in plane_vertex_membership:
                plane_vertex_membership[vi] = []
            plane_vertex_membership[vi].append((plane.normal, plane.offset))
    plane_vertex_indices = sorted(plane_vertex_membership.keys())
    n_plane = len(plane_vertex_indices)

    # Build initial x:
    #   [p_0, ..., p_{nr-1},
    #    split_mesh_0, ..., split_mesh_{ns-1},
    #    plane_v_0, ..., plane_v_{np-1}]
    split_offset = 3 * n_regions
    plane_offset = split_offset + 3 * n_split
    n_vars = 3 * n_regions + 3 * n_split + 3 * n_plane
    x0 = np.zeros(n_vars)

    # Mesh-side split copies start at resized positions
    for j, sv in enumerate(split_vertices):
        x0[split_offset + 3 * j: split_offset + 3 * j + 3] = (
            resized_positions[sv.original_index]
        )
    # Plane vertices start at their current positions
    for k, vi in enumerate(plane_vertex_indices):
        x0[plane_offset + 3 * k: plane_offset + 3 * k + 3] = (
            resized_positions[vi]
        )

    result = minimize(
        _energy_and_gradient,
        x0,
        args=(resized_positions, n_regions, region_corner_indices,
              split_vertices, flat_planes, split_plane_membership,
              plane_vertex_indices, plane_vertex_membership,
              lambda_t, lambda_p),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-7},
    )

    # Unpack result: apply rigid translation to each region's vertices
    optimized = resized_positions.copy()
    for ri in range(n_regions):
        p_i = result.x[3 * ri: 3 * ri + 3]
        for vi in region_vert_indices[ri]:
            optimized[vi] = resized_positions[vi] + p_i

    # Write back optimized plane vertex positions
    for k, vi in enumerate(plane_vertex_indices):
        optimized[vi] = result.x[plane_offset + 3 * k: plane_offset + 3 * k + 3]

    return optimized, float(result.fun)

def deform_mesh(
    mesh: o3d.geometry.TriangleMesh,
    scale: float,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    lambda_t: float = 5.0,
    lambda_p: float = 4.0,
    max_iter: int = 200,
) -> DeformationResult:
    """Deform mesh so slope regions align to integer brick grid.

    Each slope region is treated as a rigid body. The optimizer solves for translations 
    that snap region corners to integer grid coordinates while keeping split vertices together.

    Args:
        mesh: Normalized mesh (post normalize_mesh).
        scale: Optimal scale s* from compute_optimal_scale().
        assignments: (region, matched_bricks) pairs from compute_optimal_scale().
        lambda_t: Weight for topology energy (default 5).
        lambda_p: Weight for planarity energy (default 4).
        max_iter: Maximum L-BFGS-B iterations.

    Returns:
        DeformationResult with deformed vertices and metadata.
    """
    scaled_mesh = apply_scale(mesh, scale)
    flat_planes = detect_planes(scaled_mesh)

    if not assignments:
        return DeformationResult(
            deformed_vertices=np.asarray(scaled_mesh.vertices).copy(),
            split_vertices=[],
            flat_planes=flat_planes,
            slope_corner_indices=[],
            final_energy=0.0,
        )

    target_positions, split_vertices, region_vert_indices = resize_slope_regions(
        scaled_mesh, assignments,
    )

    # Identify slope corners per region for E_int
    faces = np.asarray(scaled_mesh.triangles)
    vertices_arr = np.asarray(scaled_mesh.vertices)
    corner_indices: list[int] = []
    region_corner_indices: list[list[int]] = []
    for ri, (region, _bricks) in enumerate(assignments):
        region_faces = faces[region.face_indices]
        vert_indices = np.unique(region_faces)
        region_verts = vertices_arr[vert_indices]
        if len(region_verts) == 0:
            region_corner_indices.append([])
            continue
        min_b = region_verts.min(axis=0)
        max_b = region_verts.max(axis=0)
        bb_corners = np.array([
            [min_b[0], min_b[1], min_b[2]],
            [max_b[0], min_b[1], min_b[2]],
            [min_b[0], max_b[1], min_b[2]],
            [max_b[0], max_b[1], min_b[2]],
            [min_b[0], min_b[1], max_b[2]],
            [max_b[0], min_b[1], max_b[2]],
            [min_b[0], max_b[1], max_b[2]],
            [max_b[0], max_b[1], max_b[2]],
        ])
        corners: list[int] = []
        for corner in bb_corners:
            dists = np.linalg.norm(region_verts - corner, axis=1)
            nearest = int(vert_indices[int(np.argmin(dists))])
            if nearest not in corners:
                corners.append(nearest)
        region_corner_indices.append(corners)
        corner_indices.extend(corners)

    # Optimize with rigid-body translations
    optimized_positions, final_energy = optimize_vertices(
        target_positions,
        region_vert_indices,
        region_corner_indices,
        split_vertices,
        flat_planes,
        lambda_t=lambda_t,
        lambda_p=lambda_p,
        max_iter=max_iter,
    )

    return DeformationResult(
        deformed_vertices=optimized_positions,
        split_vertices=split_vertices,
        flat_planes=flat_planes,
        slope_corner_indices=sorted(set(corner_indices)),
        final_energy=final_energy,
    )
