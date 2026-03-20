import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from mesh2brick.data.brick_library import brick_library
from mesh2brick.mesh_utils import build_face_adjacency, compute_areas, normal_angle_diff


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
    """Compute the width, height, and length of a region aligned to the slope direction.

    Args:
        mesh: The mesh.
        face_indices: List of face indices in the region.
        slope_direction: 0=+X, 1=+Y, 2=-X, 3=-Y.

    Returns:
        (length, width, height)
        length: Dimension along direction on XY plane.
        width: Dimension perpendicular to direction on XY plane.
        height: Dimension along Z axis.
    """
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


def detect_slopes(
    mesh: o3d.geometry.TriangleMesh,
    planar_deg_err: float = 10.0,
    normal_deg_err: float = 15.0,
    min_area_fraction: float = 0.1,
) -> list[SlopeRegion]:
    """Detect sloped surface regions on a mesh.

    Args:
        mesh: Normalized, Z-scaled Open3D triangle mesh.
        planar_deg_err: Faces within this angle of horizontal or vertical are excluded.
        normal_deg_err: Max angular difference for BFS grouping.
        min_area_fraction: Minimum region area as fraction of total mesh area.

    Returns:
        List of detected SlopeRegion objects.
    """
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return []

    face_areas = compute_areas(mesh)
    total_area = face_areas.sum()
    if total_area == 0:
        return []

    # Filter out planar regions
    up = np.array([0.0, 0.0, 1.0])
    cos_angles = np.abs(normals @ up)
    vert_angle_diff = np.degrees(np.arccos(np.clip(cos_angles, 0, 1)))
    is_sloped = (vert_angle_diff > planar_deg_err) & (vert_angle_diff < 90 - planar_deg_err)
    sloped_faces = set(np.where(is_sloped)[0])

    if not sloped_faces:
        return []

    # Build face adjacency for sloped faces
    adjacency = build_face_adjacency(triangles, sloped_faces)

    # BFS group connected faces with similar normals
    visited: set[int] = set()
    groups: list[list[int]] = []

    for start_face in sloped_faces:
        if start_face in visited:
            continue
        queue = deque([start_face])
        visited.add(start_face)
        group = []
        while queue:
            face = queue.popleft()
            group.append(face)
            for neighbor in adjacency.get(face, []):
                if neighbor not in visited:
                    # only compares to start_face, could potentially get unlucky and not maximize sloped area
                    if normal_angle_diff(normals[start_face], normals[neighbor]) < normal_deg_err:
                        visited.add(neighbor)
                        queue.append(neighbor)
        groups.append(group)

    # Filter by area threshold
    regions: list[SlopeRegion] = []
    for group in groups:
        group_areas = face_areas[group]
        region_area = group_areas.sum()
        if region_area < min_area_fraction * total_area:
            continue

        # Compute area-weighted average normal
        weighted_normals = normals[group] * group_areas[:, np.newaxis]
        avg_normal = weighted_normals.sum(axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm < 1e-10:
            continue
        avg_normal /= norm

        # Slope angle from horizontal = angle of the normal from vertical
        slope_angle = math.degrees(math.acos(min(abs(avg_normal[2]), 1.0)))

        # Slope direction: project normal onto XY plane, snap to cardinal
        direction = np.array([avg_normal[0], avg_normal[1]])
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction /= dir_norm
        slope_direction = _to_cardinal(direction)

        length, width, height = _compute_region_bounds(mesh, group, slope_direction)

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

    return regions


def get_slope_bricks() -> list[dict]:
    """Get all slope bricks from the brick library.

    Returns:
        List of dicts with keys: brick_id, type, length, width, height, angle.
    """
    slope_bricks = []
    for brick_id, props in brick_library.items():
        if props.get('type') != 1:
            continue
        angle = math.degrees(math.atan(props['height'] / props['length']))
        slope_bricks.append({
            'brick_id': int(brick_id),
            'length': props['length'],
            'width': props['width'],
            'height': props['height'],
            'angle': angle,
        })
    return slope_bricks


def match_slope_to_bricks(slope_angle: float, slope_bricks: list[dict] | None = None) -> list[dict]:
    """Match a detected slope angle to the best available slope bricks.

    Args:
        slope_angle: Detected angle in degrees from horizontal.
        slope_bricks: Available slope bricks (from get_slope_bricks()). If None, loads from library.

    Returns:
        List of matching bricks at the closest available angle, sorted by area (largest first).
    """
    if slope_bricks is None:
        slope_bricks = get_slope_bricks()

    if not slope_bricks:
        return []

    # Find the unique angles and pick the closest
    angles = sorted(set(b['angle'] for b in slope_bricks))
    best_angle = min(angles, key=lambda a: abs(a - slope_angle))

    # Return all bricks at that angle, sorted by area (largest first)
    matches = [b for b in slope_bricks if abs(b['angle'] - best_angle) < 0.1]
    matches.sort(key=lambda b: b['length'] * b['width'], reverse=True)
    return matches


# Backward compatibility alias
slope_to_bricks = match_slope_to_bricks


def _compute_s_min(region: SlopeRegion, slope_bricks: list[dict] | None = None) -> float | None:
    """Compute the minimum scale to fit at least one slope brick into a region.

    Returns the s_min for the easiest-to-fit (smallest) matching brick,
    or None if no brick can be matched.
    """
    matched = match_slope_to_bricks(region.slope_angle, slope_bricks)
    if not matched or region.length <= 0 or region.width <= 0:
        return None

    s_mins = []
    for brick in matched:
        s_min_b = max(brick['length'] / region.length,
                      brick['width'] / region.width)
        s_mins.append(s_min_b)
    return min(s_mins)


def compute_optimal_scale(
    regions: list[SlopeRegion],
    default_scale: float = 20.0,
    max_scale: float = 40.0,
) -> tuple[float, list[tuple[SlopeRegion, list[dict]]]]:
    """Compute the optimal global scale factor for slope brick fitting.

    For each slope region, finds matching slope bricks and computes the
    minimum scale needed to fit at least one brick. The optimal scale
    is the maximum of all per-region minimums (to achieve zero energy),
    clamped between default_scale and max_scale.

    Args:
        regions: Detected slope regions (in post-normalize_mesh coords).
        default_scale: Scale to use when no slope regions exist (also the floor).
        max_scale: Upper bound on scale to prevent huge models.

    Returns:
        (optimal_scale, assignments) where assignments is a list of
        (region, matched_bricks) pairs. Regions whose s_min > 2 * optimal_scale
        are excluded (fallback to cuboid bricks).
    """
    if not regions:
        return default_scale, []

    slope_bricks = get_slope_bricks()

    # For each region, find best matching bricks and compute s_min
    region_info: list[tuple[SlopeRegion, list[dict], float]] = []
    for region in regions:
        s_min = _compute_s_min(region, slope_bricks)
        if s_min is None:
            continue
        matched = match_slope_to_bricks(region.slope_angle, slope_bricks)
        region_info.append((region, matched, s_min))

    if not region_info:
        return default_scale, []

    # Optimal scale = max of all s_min values (zero energy), clamped
    s_star = max(info[2] for info in region_info)
    s_star = max(s_star, default_scale)
    s_star = min(s_star, max_scale)

    # Fallback: discard regions where s_star < 0.5 * s_min
    assignments: list[tuple[SlopeRegion, list[dict]]] = []
    for region, matched, s_min in region_info:
        if s_star >= 0.5 * s_min:
            assignments.append((region, matched))

    return s_star, assignments


def compute_energy(scale: float, regions: list[SlopeRegion]) -> float:
    """Compute total scale energy for a given scale. Lower is better.

    E_scale(s) = sum of max(0, s_min_i - s) for all slope regions.
    Currently only handles slopes; designed for future cylinder extension.
    """
    slope_bricks = get_slope_bricks()
    total = 0.0
    for region in regions:
        s_min = _compute_s_min(region, slope_bricks)
        if s_min is not None:
            total += max(0.0, s_min - scale)
    return total
