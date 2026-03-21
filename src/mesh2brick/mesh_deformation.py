"""Model Deformation (Grid Alignment).

Deforms a mesh so that detected slope regions snap to integer voxel grid
coordinates with dimensions that are exact multiples of slope brick sizes.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.optimize import minimize

from mesh2brick.slope_detection import SlopeRegion
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
    vertices: np.ndarray,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    position_tol: float = 1e-6,
) -> tuple[list[np.ndarray], list[int], dict[int, int], dict[int, int]]:
    """Partition vertices into slope regions and identify split vertices.

    A *split vertex* is one shared between slope-region faces and non-slope
    faces.  Sharing is detected both by index (same vertex used in both)
    and by position (GLB meshes often duplicate vertices at boundaries with
    different normals/UVs — these are "positional splits").

    For each positional split on the non-slope side, we record a mapping
    to its coincident slope-side vertex so that ``resize_slope_regions``
    and ``optimize_vertices`` can keep them in sync.

    Args:
        faces: (F, 3) triangle index array.
        vertices: (N, 3) vertex positions.
        assignments: (region, matched_bricks) pairs.
        position_tol: Max distance to consider two vertices coincident.

    Returns:
        (region_vert_indices, split_indices, vert_to_region, coincident_map):
        - region_vert_indices: unique vertex indices per region.
        - split_indices: sorted vertex indices shared between slope/non-slope.
        - vert_to_region: maps each slope vertex to its first owning region.
        - coincident_map: maps non-slope vertex index → slope vertex index
          for positional duplicates at the slope/non-slope boundary.
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

    # Index-based split vertices (shared by both slope and non-slope faces)
    split_indices_set = slope_verts & non_slope_verts

    # Position-based split vertices: non-slope vertices coincident with
    # slope-only vertices (common in GLB meshes with per-face attributes).
    slope_only = sorted(slope_verts - split_indices_set)
    non_slope_only = sorted(non_slope_verts - split_indices_set)
    coincident_map: dict[int, int] = {}  # non-slope idx → slope idx

    if slope_only and non_slope_only:
        slope_positions = vertices[slope_only]
        non_slope_positions = vertices[non_slope_only]
        # For each non-slope vertex, find closest slope vertex
        for i, ns_idx in enumerate(non_slope_only):
            dists = np.linalg.norm(slope_positions - non_slope_positions[i], axis=1)
            min_dist_idx = int(np.argmin(dists))
            if dists[min_dist_idx] < position_tol:
                s_idx = slope_only[min_dist_idx]
                coincident_map[ns_idx] = s_idx
                # Treat the slope-side vertex as a split vertex too
                split_indices_set.add(s_idx)

    split_indices = sorted(split_indices_set)

    vert_to_region: dict[int, int] = {}
    for ri, vert_arr in enumerate(region_vert_indices):
        for vi in vert_arr:
            vi_int = int(vi)
            if vi_int not in vert_to_region:
                vert_to_region[vi_int] = ri

    return region_vert_indices, split_indices, vert_to_region, coincident_map


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
) -> tuple[np.ndarray, list[SplitVertex], list[np.ndarray], dict[int, int]]:
    """Resize slope regions to exact brick-multiple dimensions.

    For each assigned region, scales its vertices so that the region's bounding box
    becomes an integer multiple of the matched brick dimensions.

    Also identifies split vertices (by index AND by position) and
    propagates the resize displacement to positional duplicates on the
    non-slope side so the mesh stays watertight.

    Args:
        mesh: Scaled mesh (post apply_scale).
        assignments: (region, matched_bricks) pairs.

    Returns:
        (target_positions, split_vertices, region_vert_indices, coincident_map):
        - target_positions: (N, 3) copy of mesh vertices with slope vertices moved.
        - split_vertices: SplitVertex objects with region_index set.
        - region_vert_indices: list of vertex index arrays, one per region.
        - coincident_map: non-slope vertex → slope vertex for positional duplicates.
    """
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    target = vertices.copy()

    region_vert_indices, split_indices, vert_to_region, coincident_map = (
        _classify_vertices(faces, vertices, assignments)
    )

    # Resize each region into a separate buffer, then average for vertices
    # that belong to multiple regions (e.g. ridge vertices between two roof
    # slopes).  Without averaging, the last region's resize overwrites
    # earlier ones, causing asymmetry.
    n_verts = len(vertices)
    region_count = np.zeros(n_verts, dtype=int)
    region_sum = np.zeros_like(vertices)

    for ri, (region, bricks) in enumerate(assignments):
        temp = vertices.copy()
        _resize_region(vertices, temp, region, bricks, region_vert_indices[ri])
        for vi in region_vert_indices[ri]:
            vi = int(vi)
            region_sum[vi] += temp[vi]
            region_count[vi] += 1

    moved = region_count > 0
    target[moved] = region_sum[moved] / region_count[moved, np.newaxis]

    # Propagate resize displacement to coincident non-slope vertices so
    # the boundary stays watertight after resizing.
    for ns_idx, s_idx in coincident_map.items():
        target[ns_idx] = target[s_idx]

    split_vertices = [
        SplitVertex(
            original_index=idx,
            region_index=vert_to_region.get(idx, 0),
            slope_var_index=-1,
            mesh_var_index=-1,
        )
        for idx in split_indices
    ]

    return target, split_vertices, region_vert_indices, coincident_map


def _energy_and_gradient(
    x: np.ndarray,
    resized_positions: np.ndarray,
    n_regions: int,
    region_corner_indices: list[list[int]],
    split_vertices: list[SplitVertex],
    split_plane_membership: dict[int, list[tuple[np.ndarray, float]]],
    plane_vertex_indices: list[int],
    plane_vertex_membership: dict[int, list[tuple[np.ndarray, float]]],
    lambda_t: float,
    lambda_p: float,
) -> tuple[float, np.ndarray]:
    """Compute E_deform = E_int + λ_t·E_topology + λ_p·E_planarity and gradient.

    Variables: region translations p_i, split mesh-side positions, plane vertex positions.

    Returns:
        (E_deform, grad) — grad has same shape as x.
    """
    n_split = len(split_vertices)
    n_plane = len(plane_vertex_indices)
    split_offset = 3 * n_regions
    plane_offset = split_offset + 3 * n_split
    n_vars = 3 * n_regions + 3 * n_split + 3 * n_plane

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

    # E_planarity (Eq 10): keep split mesh-side copies on their flat planes
    for j, sv in enumerate(split_vertices):
        for n_i, v_0i in split_plane_membership.get(sv.original_index, []):
            v_mesh = x[split_offset + 3 * j: split_offset + 3 * j + 3]
            proj = float(np.dot(n_i, v_mesh)) - v_0i
            E_deform += lambda_p * proj * proj
            grad[split_offset + 3 * j: split_offset + 3 * j + 3] += (
                lambda_p * 2.0 * proj * n_i
            )

    # E_planarity for plane vertices (non-slope vertices on detected planes)
    for k, vi in enumerate(plane_vertex_indices):
        v_pos = x[plane_offset + 3 * k: plane_offset + 3 * k + 3]
        for n_i, v_0i in plane_vertex_membership[vi]:
            proj = float(np.dot(n_i, v_pos)) - v_0i
            E_deform += lambda_p * proj * proj
            grad[plane_offset + 3 * k: plane_offset + 3 * k + 3] += (
                lambda_p * 2.0 * proj * n_i
            )

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
    """Minimize E_deform = E_int + λ_t·E_topology + λ_p·E_planarity.

    Variables:
    - p_i: rigid translations per slope region
    - Split vertex mesh-side positions
    - Plane vertex positions (non-slope vertices on detected flat planes)

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
    n_verts = len(resized_positions)

    if n_regions == 0:
        return resized_positions.copy(), 0.0

    # Collect all slope vertex indices and build region lookup
    slope_verts: set[int] = set()
    vert_to_regions: dict[int, list[int]] = {}
    for ri, vert_arr in enumerate(region_vert_indices):
        for v in vert_arr:
            vi = int(v)
            slope_verts.add(vi)
            vert_to_regions.setdefault(vi, []).append(ri)
    split_vert_set = {sv.original_index for sv in split_vertices}

    # Precompute plane membership for split vertices
    split_plane_membership: dict[int, list[tuple[np.ndarray, float]]] = {}
    for plane in flat_planes:
        vs = set(plane.vertex_indices)
        for sv in split_vertices:
            if sv.original_index in vs:
                split_plane_membership.setdefault(sv.original_index, []).append(
                    (plane.normal, plane.offset)
                )

    # Plane vertices: non-slope vertices on detected flat planes
    plane_vertex_membership: dict[int, list[tuple[np.ndarray, float]]] = {}
    for plane in flat_planes:
        for vi in plane.vertex_indices:
            if vi not in slope_verts:
                plane_vertex_membership.setdefault(vi, []).append(
                    (plane.normal, plane.offset)
                )
    plane_vertex_indices = sorted(plane_vertex_membership.keys())
    n_plane = len(plane_vertex_indices)

    # Build initial x:
    #   [p_0..p_{nr-1}, split_mesh_0..split_mesh_{ns-1}, plane_v_0..plane_v_{np-1}]
    split_offset = 3 * n_regions
    plane_offset = split_offset + 3 * n_split
    n_vars = 3 * n_regions + 3 * n_split + 3 * n_plane
    x0 = np.zeros(n_vars)

    for j, sv in enumerate(split_vertices):
        x0[split_offset + 3 * j: split_offset + 3 * j + 3] = (
            resized_positions[sv.original_index]
        )
    for k, vi in enumerate(plane_vertex_indices):
        x0[plane_offset + 3 * k: plane_offset + 3 * k + 3] = (
            resized_positions[vi]
        )

    result = minimize(
        _energy_and_gradient,
        x0,
        args=(resized_positions, n_regions, region_corner_indices,
              split_vertices, split_plane_membership,
              plane_vertex_indices, plane_vertex_membership,
              lambda_t, lambda_p),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-7},
    )

    # --- Unpack result ---
    optimized = resized_positions.copy()

    # Slope vertices: rigid translations (average for multi-region)
    translation_sum = np.zeros((n_verts, 3))
    translation_count = np.zeros(n_verts, dtype=int)
    for ri in range(n_regions):
        p_i = result.x[3 * ri: 3 * ri + 3]
        for vi in region_vert_indices[ri]:
            vi = int(vi)
            translation_sum[vi] += p_i
            translation_count[vi] += 1

    has_translation = translation_count > 0
    avg_translation = np.zeros((n_verts, 3))
    avg_translation[has_translation] = (
        translation_sum[has_translation]
        / translation_count[has_translation, np.newaxis]
    )
    optimized[has_translation] = (
        resized_positions[has_translation] + avg_translation[has_translation]
    )

    # Split vertices: average slope-side and mesh-side
    for j, sv in enumerate(split_vertices):
        ri = sv.region_index
        p_i = result.x[3 * ri: 3 * ri + 3]
        slope_side = resized_positions[sv.original_index] + p_i
        mesh_side = result.x[split_offset + 3 * j: split_offset + 3 * j + 3]
        optimized[sv.original_index] = 0.5 * (slope_side + mesh_side)

    # Plane vertices: direct from optimizer
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

    target_positions, split_vertices, region_vert_indices, coincident_map = (
        resize_slope_regions(scaled_mesh, assignments)
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

    # Sync coincident non-slope vertices with their slope-side counterparts.
    # These are positional duplicates (same position, different index) that
    # must track the slope vertex they're paired with.
    for ns_idx, s_idx in coincident_map.items():
        optimized_positions[ns_idx] = optimized_positions[s_idx]

    return DeformationResult(
        deformed_vertices=optimized_positions,
        split_vertices=split_vertices,
        flat_planes=flat_planes,
        slope_corner_indices=sorted(set(corner_indices)),
        final_energy=final_energy,
    )
