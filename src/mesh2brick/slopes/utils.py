import math
from collections import defaultdict

import numpy as np
import open3d as o3d


def build_face_adjacency(triangles: np.ndarray, face_indices: set[int]) -> dict[int, list[int]]:
    """Build adjacency graph for faces sharing an edge, restricted to face_indices."""

    # Map edges (key) to list of faces (value) that contain that edge
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f_idx in face_indices:
        tri = triangles[f_idx]
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            edge = (min(tri[i], tri[j]), max(tri[i], tri[j]))
            edge_to_faces[edge].append(f_idx)

    # Build adjacency list to represent graph of faces sharing an edge
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
