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
from scipy.spatial import cKDTree

from .detection import SlopeRegion, Plane
from .utils import build_face_adjacency, normal_angle_diff


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
    new_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(new_mesh.vertices)
    vertices *= scale
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return new_mesh


def _classify_vertices(
    faces: np.ndarray,
    vertices: np.ndarray,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    position_tol: float = 1e-6,
) -> tuple[list[np.ndarray], list[int], dict[int, int], dict[int, int]]:
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

    split_indices_set = slope_verts & non_slope_verts

    slope_only = sorted(slope_verts - split_indices_set)
    non_slope_only = sorted(non_slope_verts - split_indices_set)
    coincident_map: dict[int, int] = {} 

    if slope_only and non_slope_only:
        slope_positions = vertices[slope_only]
        non_slope_positions = vertices[non_slope_only]
        for i, ns_idx in enumerate(non_slope_only):
            dists = np.linalg.norm(slope_positions - non_slope_positions[i], axis=1)
            min_dist_idx = int(np.argmin(dists))
            if dists[min_dist_idx] < position_tol:
                s_idx = slope_only[min_dist_idx]
                coincident_map[ns_idx] = s_idx
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
    # Snap width (lateral axis) independently
    if region_dims[width_axis] > 0:
        target_w = max(1, round(region_dims[width_axis] / brick['width'])) * brick['width']
        scale_factors[width_axis] = target_w / region_dims[width_axis]

    # Snap length to grid, then derive height to enforce the brick's angle
    # For h=3 slopes, the stud column overlaps between steps, so advance = run
    if region_dims[length_axis] > 0:
        run = brick['length'] - 1 if brick['height'] == 3 and brick['length'] > 1 else brick['length']
        n_steps = max(1, round(region_dims[length_axis] / run))
        target_l = (n_steps - 1) * run + brick['length']
        scale_factors[length_axis] = target_l / region_dims[length_axis]

        if region_dims[height_axis] > 0:
            target_h = n_steps * brick['height'] / 3.0
            scale_factors[height_axis] = target_h / region_dims[height_axis]

    for vi in vert_indices:
        target[vi] = centroid + (vertices[vi] - centroid) * scale_factors


def resize_slope_regions(
    mesh: o3d.geometry.TriangleMesh,
    assignments: list[tuple[SlopeRegion, list[dict]]],
) -> tuple[np.ndarray, list[SplitVertex], list[np.ndarray], dict[int, int]]:
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    target = vertices.copy()

    region_vert_indices, split_indices, vert_to_region, coincident_map = (
        _classify_vertices(faces, vertices, assignments)
    )

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

    # Snap near-coincident ridge vertices between different slope regions.
    # The mesh often has separate vertex indices at the ridge; independent
    # centroid-based scaling pushes them apart.  Average their positions.
    vert_to_regions: dict[int, set[int]] = {}
    for ri, vert_arr in enumerate(region_vert_indices):
        for v in vert_arr:
            vert_to_regions.setdefault(int(v), set()).add(ri)
    all_slope_verts = sorted(vert_to_regions.keys())
    if len(all_slope_verts) > 1:
        orig_positions = vertices[all_slope_verts]
        tree = cKDTree(orig_positions)
        pairs = tree.query_pairs(r=0.5)
        # Union-find to group coincident vertices
        parent: dict[int, int] = {}
        def _find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        for i, j in pairs:
            vi, vj = all_slope_verts[i], all_slope_verts[j]
            # Only link if they belong to different regions
            if vert_to_regions[vi] != vert_to_regions[vj]:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(all_slope_verts)):
            groups[_find(idx)].append(idx)
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue
            vert_ids = [all_slope_verts[i] for i in group_indices]
            avg_pos = np.mean(target[vert_ids], axis=0)
            for vi in vert_ids:
                target[vi] = avg_pos

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
    n_split = len(split_vertices)
    n_plane = len(plane_vertex_indices)
    split_offset = 3 * n_regions
    plane_offset = split_offset + 3 * n_split
    n_vars = 3 * n_regions + 3 * n_split + 3 * n_plane

    grad = np.zeros(n_vars)
    E_deform = 0.0

    for ri in range(n_regions):
        p_ri = x[3 * ri: 3 * ri + 3]
        for vi in region_corner_indices[ri]:
            v_i = resized_positions[vi] + p_ri
            diff = v_i - np.round(v_i)
            E_deform += np.dot(diff, diff)
            grad[3 * ri: 3 * ri + 3] += 2.0 * diff

    for j, sv in enumerate(split_vertices):
        ri = sv.region_index
        p_ri = x[3 * ri: 3 * ri + 3]
        v_slope = resized_positions[sv.original_index] + p_ri
        v_mesh = x[split_offset + 3 * j: split_offset + 3 * j + 3]
        diff = v_slope - v_mesh
        E_deform += lambda_t * np.dot(diff, diff)
        grad[3 * ri: 3 * ri + 3] += lambda_t * 2.0 * diff
        grad[split_offset + 3 * j: split_offset + 3 * j + 3] += lambda_t * (-2.0 * diff)

    for j, sv in enumerate(split_vertices):
        for n_i, v_0i in split_plane_membership.get(sv.original_index, []):
            v_mesh = x[split_offset + 3 * j: split_offset + 3 * j + 3]
            proj = float(np.dot(n_i, v_mesh)) - v_0i
            E_deform += lambda_p * proj * proj
            grad[split_offset + 3 * j: split_offset + 3 * j + 3] += (
                lambda_p * 2.0 * proj * n_i
            )

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
    n_regions = len(region_vert_indices)
    n_split = len(split_vertices)
    n_verts = len(resized_positions)

    if n_regions == 0:
        return resized_positions.copy(), 0.0

    slope_verts: set[int] = set()
    vert_to_regions: dict[int, list[int]] = {}
    for ri, vert_arr in enumerate(region_vert_indices):
        for v in vert_arr:
            vi = int(v)
            slope_verts.add(vi)
            vert_to_regions.setdefault(vi, []).append(ri)

    split_plane_membership: dict[int, list[tuple[np.ndarray, float]]] = {}
    for plane in flat_planes:
        vs = set(plane.vertex_indices)
        for sv in split_vertices:
            if sv.original_index in vs:
                split_plane_membership.setdefault(sv.original_index, []).append(
                    (plane.normal, plane.offset)
                )

    plane_vertex_membership: dict[int, list[tuple[np.ndarray, float]]] = {}
    for plane in flat_planes:
        for vi in plane.vertex_indices:
            if vi not in slope_verts:
                plane_vertex_membership.setdefault(vi, []).append(
                    (plane.normal, plane.offset)
                )
    plane_vertex_indices = sorted(plane_vertex_membership.keys())
    n_plane = len(plane_vertex_indices)

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

    optimized = resized_positions.copy()

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

    for j, sv in enumerate(split_vertices):
        ri = sv.region_index
        p_i = result.x[3 * ri: 3 * ri + 3]
        slope_side = resized_positions[sv.original_index] + p_i
        mesh_side = result.x[split_offset + 3 * j: split_offset + 3 * j + 3]
        optimized[sv.original_index] = 0.5 * (slope_side + mesh_side)

    for k, vi in enumerate(plane_vertex_indices):
        optimized[vi] = result.x[plane_offset + 3 * k: plane_offset + 3 * k + 3]

    return optimized, float(result.fun)


def deform_mesh(
    mesh: o3d.geometry.TriangleMesh,
    scale: float,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    flat_planes: list[Plane],
    lambda_t: float = 5.0,
    lambda_p: float = 4.0,
    max_iter: int = 200,
) -> DeformationResult:
    scaled_mesh = apply_scale(mesh, scale)
    
    # Scale plane offsets natively to work on the scaled voxel grid
    for plane in flat_planes:
        plane.offset *= scale

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

    for ns_idx, s_idx in coincident_map.items():
        optimized_positions[ns_idx] = optimized_positions[s_idx]

    return DeformationResult(
        deformed_vertices=optimized_positions,
        split_vertices=split_vertices,
        flat_planes=flat_planes,
        slope_corner_indices=sorted(set(corner_indices)),
        final_energy=final_energy,
    )
