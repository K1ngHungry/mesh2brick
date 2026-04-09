import math
from collections import defaultdict

import numpy as np
import open3d as o3d


def build_face_adjacency(triangles: np.ndarray, face_indices: set[int]) -> dict[int, list[int]]:
    """Build adjacency graph for faces sharing an edge, restricted to face_indices.

    Maps each edge (sorted vertex pair) to the faces that contain it, then
    connects all faces that share an edge into an adjacency list.
    """
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f_idx in face_indices:
        tri = triangles[f_idx]
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            edge = (min(tri[i], tri[j]), max(tri[i], tri[j]))
            edge_to_faces[edge].append(f_idx)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for f_indices in edge_to_faces.values():
        for i in range(len(f_indices)):
            for j in range(i + 1, len(f_indices)):
                adjacency[f_indices[i]].append(f_indices[j])
                adjacency[f_indices[j]].append(f_indices[i])
    return adjacency


def compute_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Compute the area of each triangle in the mesh.
    Formula: Area = 0.5 * || (v1 - v0) x (v2 - v0) ||
    where ||...|| is the L2 norm (magnitude) and 'x' is the cross product.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def normal_angle_diff(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle between two unit normals in degrees."""
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return math.degrees(math.acos(abs(dot)))


_AXIS_NORMALS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=float)


def to_axis(normal: np.ndarray, planar_err: float) -> np.ndarray | None:
    """If *normal* is within *planar_err* of an axis direction, return
    that axis direction; otherwise return None."""
    dots = _AXIS_NORMALS @ normal
    best_idx = int(np.argmax(dots))
    best_dot = dots[best_idx]
    angle = math.degrees(math.acos(np.clip(best_dot, -1.0, 1.0)))
    if angle <= planar_err:
        return _AXIS_NORMALS[best_idx].copy()
    return None


def to_cardinal(direction: np.ndarray) -> int:
    """Snap a 2D direction to the nearest cardinal axis.
    Returns 0=+X, 1=+Y, 2=-X, 3=-Y."""
    candidates = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    dots = candidates @ direction
    return int(np.argmax(dots))


def slope_run(length: int, height: int) -> int:
    """Effective horizontal advance per staircase step.

    For h=3 slope bricks with length > 1, the top stud column overlaps
    the next step, so the run is length - 1.  All other bricks advance
    by their full length.
    """
    return length - 1 if height == 3 and length > 1 else length


def compute_region_bounds(
    mesh: o3d.geometry.TriangleMesh,
    face_indices: list[int],
    slope_direction: int,
) -> tuple[float, float, float]:
    """Compute (length, width, height) of a mesh region's bounding box.

    Length is along the slope direction axis, width is the lateral axis,
    and height is the Z extent.
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

    if slope_direction in (0, 2):  # X-aligned slope
        length = ranges[0]
        width = ranges[1]
    else:  # Y-aligned slope
        length = ranges[1]
        width = ranges[0]

    return length, width, height
