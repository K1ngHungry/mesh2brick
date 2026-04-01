import functools
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from mesh2brick.data.brick_library import brick_library
from .utils import build_face_adjacency, compute_areas, normal_angle_diff


@dataclass
class SlopeRegion:
    """A connected region of mesh faces sharing a similar sloped normal."""
    face_indices: list[int]
    avg_normal: np.ndarray
    area: float
    slope_angle: float       # degrees from horizontal
    slope_direction: int     # 0=+X, 1=+Y, 2=-X, 3=-Y
    length: float = 0.0      # Horizontal dimension along slope direction
    width: float = 0.0       
    height: float = 0.0      

@dataclass
class Plane:
    """A detected flat (axis-aligned) surface region."""
    face_indices: list[int]
    vertex_indices: list[int]
    normal: np.ndarray
    offset: float

@dataclass
class Features:
    """Output of feature detection containing slopes and planes."""
    regions: list[SlopeRegion]
    planes: list[Plane]

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

def _to_cardinal(direction: np.ndarray) -> int:
    """Snap a 2D direction to the nearest cardinal axis.
    Returns 0=+X, 1=+Y, 2=-X, 3=-Y."""
    candidates = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    dots = candidates @ direction
    return int(np.argmax(dots))

def _compute_region_bounds(
    mesh: o3d.geometry.TriangleMesh,
    face_indices: list[int],
    slope_direction: int
) -> tuple[float, float, float]:
    triangles = np.asarray(mesh.triangles)[face_indices]
    vertex_indices = np.unique(triangles)
    vertices = np.asarray(mesh.vertices)[vertex_indices]

    if len(vertices) == 0:
        return 0.0, 0.0, 0.0

    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    ranges = max_bound - min_bound  # [dx, dy, dz]

    height = ranges[2]

    # slope_direction: 0=+X, 1=+Y, 2=-X, 3=-Y
    if slope_direction == 0 or slope_direction == 2:  # X-aligned slope
        length = ranges[0]
        width = ranges[1]
    else:  # Y-aligned slope
        length = ranges[1]
        width = ranges[0]

    return length, width, height

def detect_features(
    mesh: o3d.geometry.TriangleMesh,
    planar_deg_err: float = 10.0,
    normal_deg_err: float = 10.0,
    min_area_fraction: float = 0.01,
    min_plane_faces: int = 3,
) -> Features:
    """Detect both sloped surface regions and flat axis-aligned planes on a mesh.

    Args:
        mesh: Normalized, Z-scaled Open3D triangle mesh.
        planar_deg_err: Faces within this angle of horizontal or vertical are planes, excluded from slopes.
        normal_deg_err: Max angular difference for BFS grouping of slopes.
        min_area_fraction: Minimum slope region area as fraction of total mesh area.
        min_plane_faces: Minimum faces to form a plane group.

    Returns:
        Features dataclass with regions and planes.
    """
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    if len(triangles) == 0:
        return Features([], [])

    face_areas = compute_areas(mesh)
    total_area = face_areas.sum()
    if total_area == 0:
        return Features([], [])

    # Filter out planar regions
    cos_angles = np.abs(normals[:, 2])
    vert_angle_diff = np.degrees(np.arccos(np.clip(cos_angles, 0, 1)))

    flat_faces: dict[int, np.ndarray] = {}
    for fi in range(len(triangles)):
        axis = _to_axis(normals[fi], planar_deg_err)
        if axis is not None:
            flat_faces[fi] = axis

    is_sloped = (vert_angle_diff > planar_deg_err) & (vert_angle_diff < 90 - planar_deg_err)
    sloped_faces = set(np.where(is_sloped)[0]) - set(flat_faces.keys())

    # --- 1. DETECT SLOPES ---
    if sloped_faces:
        adjacency_slopes = build_face_adjacency(triangles, sloped_faces)
        visited_slopes: set[int] = set()
        slope_groups: list[list[int]] = []

        for start_face in sloped_faces:
            if start_face in visited_slopes:
                continue
            queue = deque([start_face])
            visited_slopes.add(start_face)
            group = []
            while queue:
                face = queue.popleft()
                group.append(face)
                for neighbor in adjacency_slopes.get(face, []):
                    if neighbor not in visited_slopes:
                        if normal_angle_diff(normals[start_face], normals[neighbor]) < normal_deg_err:
                            visited_slopes.add(neighbor)
                            queue.append(neighbor)
            slope_groups.append(group)
    else:
        slope_groups = []

    regions: list[SlopeRegion] = []
    for group in slope_groups:
        group_areas = face_areas[group]
        region_area = group_areas.sum()
        if region_area < min_area_fraction * total_area:
            continue

        weighted_normals = normals[group] * group_areas[:, np.newaxis]
        avg_normal = weighted_normals.sum(axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm < 1e-10:
            continue
        avg_normal /= norm

        # Skip downward-facing slopes (bottom/interior surfaces)
        # These create duplicate opposing regions that cause brick overlaps
        if avg_normal[2] < 0:
            print(f"  [FILTERED] Downward slope: nz={avg_normal[2]:.3f}, angle={math.degrees(math.acos(min(abs(avg_normal[2]), 1.0))):.1f}°")
            continue

        slope_angle = math.degrees(math.acos(min(abs(avg_normal[2]), 1.0)))
        print(f"  Region: nz={avg_normal[2]:.3f}, angle={slope_angle:.1f}°, dir={_to_cardinal(np.array([avg_normal[0], avg_normal[1]]))}")

        direction = np.array([avg_normal[0], avg_normal[1]])
        if np.linalg.norm(direction) < 1e-10:
            continue
        slope_direction = _to_cardinal(direction)

        length, width, height = _compute_region_bounds(mesh, group, slope_direction)

        # Skip extremely tiny bevels/noise that barely exist in footprint area
        if length < 0.05 or width < 0.05:
            continue

        regions.append(SlopeRegion(
            face_indices=group,
            avg_normal=avg_normal,
            area=region_area,
            slope_angle=slope_angle,
            slope_direction=slope_direction,
            length=length,
            width=width,
            height=height,
        ))

    # --- 2. DETECT PLANES ---
    if flat_faces:
        adjacency_planes = build_face_adjacency(triangles, set(flat_faces.keys()))
        visited_planes: set[int] = set()
        plane_groups: list[tuple[list[int], np.ndarray]] = []

        for start_face in flat_faces:
            if start_face in visited_planes:
                continue
            seed_axis = flat_faces[start_face]

            queue = deque([start_face])
            visited_planes.add(start_face)
            group = []
            while queue:
                face = queue.popleft()
                group.append(face)
                for neighbor in adjacency_planes.get(face, []):
                    if neighbor not in visited_planes:
                        if np.array_equal(flat_faces[neighbor], seed_axis):
                            visited_planes.add(neighbor)
                            queue.append(neighbor)

            if len(group) >= min_plane_faces:
                plane_groups.append((group, seed_axis))
    else:
        plane_groups = []

    planes: list[Plane] = []
    centroids = vertices[triangles].mean(axis=1) # (N, 3)
    for face_list, axis_normal in plane_groups:
        vert_indices = np.unique(triangles[face_list]).tolist()
        group_centroids = centroids[face_list]
        offset = float(np.dot(axis_normal, group_centroids.mean(axis=0)))
        planes.append(Plane(
            face_indices=face_list,
            vertex_indices=vert_indices,
            normal=axis_normal,
            offset=offset,
        ))

    return Features(regions=regions, planes=planes)


@functools.cache
def get_slope_bricks() -> list[dict]:
    """Get all slope bricks from the brick library.

    Returns:
        List of dicts with keys: brick_id, type, length, width, height, angle.
    """
    slope_bricks = []
    for brick_id, props in brick_library.items():
        if props.get('type') != 1:
            continue
        # h=3 slopes have a stud on top, so effective run is length-1
        run = props['length'] - 1 if props['height'] == 3 and props['length'] > 1 else props['length']
        angle = math.degrees(math.atan(props['height'] / run))
        slope_bricks.append({
            'brick_id': int(brick_id),
            'length': props['length'],
            'width': props['width'],
            'height': props['height'],
            'run': run,
            'angle': angle,
        })
    return slope_bricks

def match_slope_to_bricks(slope_angle: float, slope_bricks: list[dict] | None = None) -> list[dict]:
    if slope_bricks is None:
        slope_bricks = get_slope_bricks()

    if not slope_bricks:
        return []

    angles = sorted(set(b['angle'] for b in slope_bricks))
    best_angle = min(angles, key=lambda a: abs(a - slope_angle))

    matches = [b for b in slope_bricks if abs(b['angle'] - best_angle) < 0.1]
    matches.sort(key=lambda b: b['length'] * b['width'], reverse=True)
    return matches

def iso_to_voxel_angle(iso_angle: float, z_scale: float = 3.0) -> float:
    return math.degrees(math.atan(z_scale * math.tan(math.radians(iso_angle))))

def compute_optimal_scale(
    regions: list[SlopeRegion],
    default_scale: float = 20.0,
    max_scale: float = 40.0,
    min_steps: int = 2,
) -> tuple[float, list[tuple[SlopeRegion, list[dict]]]]:
    if not regions:
        return default_scale, []

    slope_bricks = get_slope_bricks()

    region_info: list[tuple[SlopeRegion, list[dict], float]] = []
    for region in regions:
        voxel_angle = iso_to_voxel_angle(region.slope_angle)
        matched = match_slope_to_bricks(voxel_angle, slope_bricks)
        if not matched or region.length <= 0 or region.width <= 0:
            continue

        s_min = min(
            max(min_steps * b['length'] / region.length, b['width'] / region.width)
            for b in matched
        )
        region_info.append((region, matched, s_min))

    if not region_info:
        return default_scale, []

    s_star = max(info[2] for info in region_info)
    s_star = max(s_star, default_scale)
    s_star = min(s_star, max_scale)

    assignments: list[tuple[SlopeRegion, list[dict]]] = []
    for region, matched, s_min in region_info:
        if s_star >= 0.5 * s_min:
            assignments.append((region, matched))

    return s_star, assignments
