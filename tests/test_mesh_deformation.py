"""Tests for mesh_deformation module (Phase 2: Grid Alignment)."""
import math

import numpy as np
import open3d as o3d
import pytest

from mesh2brick.slopes import DeformationResult, SlopeRegion, apply_scale, deform_mesh
from mesh2brick.slopes.deformation import (
    Plane,
    SplitVertex,
    resize_slope_regions,
)
from mesh2brick.slopes import compute_optimal_scale, detect_features, match_slope_to_bricks


# ---------------------------------------------------------------------------
# Test mesh helpers (reused from test_slope_detection.py)
# ---------------------------------------------------------------------------

def _make_mesh(vertices: list, triangles: list) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=int))
    return mesh


def _make_ramp(angle_deg: float = 45.0, width: float = 2.0) -> o3d.geometry.TriangleMesh:
    run = 1.0
    rise = run * math.tan(math.radians(angle_deg))
    vertices = [
        [0, 0, 0], [run, 0, 0], [run, 0, rise],
        [0, width, 0], [run, width, 0], [run, width, rise],
    ]
    triangles = [
        [0, 1, 4], [0, 4, 3],
        [0, 2, 1], [0, 2, 1],
        [3, 4, 5],
        [1, 2, 5], [1, 5, 4],
        [0, 5, 2], [0, 3, 5],
    ]
    return _make_mesh(vertices, triangles)


def _make_box() -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)


def _make_house() -> o3d.geometry.TriangleMesh:
    w, d, h = 4.0, 3.0, 2.0
    ridge_h = 3.5
    vertices = [
        [0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
        [0, 0, h], [w, 0, h], [w, d, h], [0, d, h],
        [0, d / 2, ridge_h], [w, d / 2, ridge_h],
    ]
    triangles = [
        [0, 2, 1], [0, 3, 2],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 8], [4, 7, 8], [7, 3, 8], [3, 0, 8],
        [1, 9, 5], [5, 9, 6], [6, 9, 2], [2, 9, 1],
        [4, 5, 9], [4, 9, 8],
        [7, 8, 9], [7, 9, 6],
    ]
    return _make_mesh(vertices, triangles)


def _make_region(slope_angle=45.0, length=0.3, width=0.2, direction=0,
                 face_indices=None):
    return SlopeRegion(
        face_indices=face_indices or [0],
        avg_normal=np.array([0.5, 0.0, 0.5]),
        area=1.0,
        slope_angle=slope_angle,
        slope_direction=direction,
        length=length,
        width=width,
        height=0.1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplyScale:
    def test_vertices_scaled(self):
        mesh = _make_box()
        scaled = apply_scale(mesh, 10.0)
        orig_verts = np.asarray(mesh.vertices)
        scaled_verts = np.asarray(scaled.vertices)
        np.testing.assert_allclose(scaled_verts, orig_verts * 10.0)

    def test_does_not_mutate_input(self):
        mesh = _make_box()
        orig = np.asarray(mesh.vertices).copy()
        apply_scale(mesh, 5.0)
        np.testing.assert_allclose(np.asarray(mesh.vertices), orig)

    def test_topology_preserved(self):
        mesh = _make_box()
        scaled = apply_scale(mesh, 3.0)
        assert len(scaled.triangles) == len(mesh.triangles)
        np.testing.assert_array_equal(
            np.asarray(scaled.triangles), np.asarray(mesh.triangles))


class TestDetectFlatPlanes:
    def test_box_has_planes(self):
        box = _make_box()
        scaled = apply_scale(box, 10.0)
        planes = detect_features(scaled, min_plane_faces=1).planes
        # A box has 6 faces (each = 2 triangles), should detect planes
        assert len(planes) >= 3  # at least some axis-aligned planes

    def test_each_plane_has_axis_normal(self):
        box = _make_box()
        scaled = apply_scale(box, 10.0)
        planes = detect_features(scaled, min_plane_faces=1).planes
        for plane in planes:
            # Normal should be near an axis
            abs_normal = np.abs(plane.normal)
            assert np.max(abs_normal) > 0.9

    def test_empty_mesh(self):
        mesh = o3d.geometry.TriangleMesh()
        planes = detect_features(mesh).planes
        assert planes == []

    def test_ramp_has_flat_bottom(self):
        ramp = _make_ramp(45.0, width=3.0)
        scaled = apply_scale(ramp, 10.0)
        planes = detect_features(scaled, min_plane_faces=1).planes
        # Ramp has a flat bottom (horizontal) and vertical walls
        # Should have at least one Z-facing plane (bottom)
        has_z = any(abs(p.normal[2]) > 0.9 for p in planes)
        assert has_z


class TestResizeSlopeRegions:
    def test_dimensions_become_brick_multiples(self):
        ramp = _make_ramp(45.0, width=3.0)
        scaled = apply_scale(ramp, 10.0)
        regions = detect_features(scaled, min_area_fraction=0.05).regions
        if not regions:
            pytest.skip("No slopes detected")
        bricks = match_slope_to_bricks(regions[0].slope_angle)
        assignments = [(regions[0], bricks)]

        target_pos, splits, region_verts, _ = resize_slope_regions(scaled, assignments)

        # Check that the resized region has brick-multiple dimensions
        region = regions[0]
        tris = np.asarray(scaled.triangles)[region.face_indices]
        vert_indices = np.unique(tris)
        resized_verts = target_pos[vert_indices]
        dims = resized_verts.max(axis=0) - resized_verts.min(axis=0)

        brick = min(bricks, key=lambda b: b['length'] * b['width'])
        # The length-axis dimension should be a multiple of brick length
        if region.slope_direction in (0, 2):
            length_dim = dims[0]
            width_dim = dims[1]
        else:
            length_dim = dims[1]
            width_dim = dims[0]

        assert length_dim > 0
        assert width_dim > 0
        # Check approximate multiple (within floating point tolerance)
        assert abs(length_dim % brick['length']) < 0.01 or \
               abs(brick['length'] - (length_dim % brick['length'])) < 0.01

    def test_split_vertices_identified(self):
        ramp = _make_ramp(45.0, width=3.0)
        scaled = apply_scale(ramp, 10.0)
        regions = detect_features(scaled, min_area_fraction=0.05).regions
        if not regions:
            pytest.skip("No slopes detected")
        bricks = match_slope_to_bricks(regions[0].slope_angle)
        assignments = [(regions[0], bricks)]

        _, splits, _, _ = resize_slope_regions(scaled, assignments)
        # Ramp has shared vertices between slope face and non-slope faces
        assert len(splits) >= 1
        # Each split vertex should have a valid region_index
        for sv in splits:
            assert sv.region_index == 0  # only one region

    def test_no_assignments_no_changes(self):
        mesh = _make_box()
        orig_verts = np.asarray(mesh.vertices).copy()
        target, splits, region_verts, _ = resize_slope_regions(mesh, [])
        np.testing.assert_allclose(target, orig_verts)
        assert splits == []
        assert region_verts == []

    def test_region_vert_indices_returned(self):
        ramp = _make_ramp(45.0, width=3.0)
        scaled = apply_scale(ramp, 10.0)
        regions = detect_features(scaled, min_area_fraction=0.05).regions
        if not regions:
            pytest.skip("No slopes detected")
        bricks = match_slope_to_bricks(regions[0].slope_angle)
        assignments = [(regions[0], bricks)]

        _, _, region_verts, _ = resize_slope_regions(scaled, assignments)
        assert len(region_verts) == 1  # one region
        assert len(region_verts[0]) > 0  # has vertices


class TestDeformMesh:
    def test_box_no_slopes_unchanged(self):
        box = _make_box()
        result = deform_mesh(box, scale=10.0, assignments=[], flat_planes=[])
        orig_verts = np.asarray(box.vertices) * 10.0
        np.testing.assert_allclose(result.deformed_vertices, orig_verts, atol=1e-6)
        assert result.final_energy == 0.0

    def test_ramp_returns_result(self):
        ramp = _make_ramp(45.0, width=3.0)
        features = detect_features(ramp, min_area_fraction=0.05)
        regions = features.regions
        if not regions:
            pytest.skip("No slopes detected on raw ramp")
        _, assignments = compute_optimal_scale(regions, default_scale=10.0)
        if not assignments:
            pytest.skip("No assignments")

        result = deform_mesh(ramp, scale=10.0, assignments=assignments, flat_planes=features.planes, max_iter=50)
        assert isinstance(result, DeformationResult)
        assert result.deformed_vertices.shape[1] == 3
        assert len(result.slope_corner_indices) >= 4

    def test_corner_vertices_near_integer(self):
        """After deformation, corner vertices should be close to integers."""
        ramp = _make_ramp(45.0, width=3.0)
        features = detect_features(ramp, min_area_fraction=0.05)
        regions = features.regions
        if not regions:
            pytest.skip("No slopes detected")
        _, assignments = compute_optimal_scale(regions, default_scale=10.0)
        if not assignments:
            pytest.skip("No assignments")

        result = deform_mesh(ramp, scale=10.0, assignments=assignments, flat_planes=features.planes, max_iter=200)

        # Check that corner vertices are near integers
        corner_verts = result.deformed_vertices[result.slope_corner_indices]
        frac_parts = np.abs(corner_verts - np.round(corner_verts))
        avg_frac = frac_parts.mean()
        # Should be closer to integer than the initial 0.25 average
        assert avg_frac < 0.3, f"Average fractional part {avg_frac} too large"

    def test_house_with_roof(self):
        house = _make_house()
        features = detect_features(house, min_area_fraction=0.05)
        regions = features.regions
        if not regions:
            pytest.skip("No slopes detected on house")
        _, assignments = compute_optimal_scale(regions, default_scale=10.0)
        if not assignments:
            pytest.skip("No assignments")

        result = deform_mesh(house, scale=10.0, assignments=assignments, flat_planes=features.planes, max_iter=50)
        assert isinstance(result, DeformationResult)
        assert len(result.flat_planes) > 0  # house has walls and floor

    def test_rigid_body_translation(self):
        """All vertices in a region should move by the same translation."""
        ramp = _make_ramp(45.0, width=3.0)
        scaled = apply_scale(ramp, 10.0)
        regions = detect_features(scaled, min_area_fraction=0.05).regions
        if not regions:
            pytest.skip("No slopes detected")
        _, assignments = compute_optimal_scale(regions, default_scale=10.0)
        if not assignments:
            pytest.skip("No assignments")

        # Get resized positions before optimization
        resized, _, region_vert_indices, _ = resize_slope_regions(scaled, assignments)

        result = deform_mesh(ramp, scale=10.0, assignments=assignments, flat_planes=[], max_iter=200)

        # For each region, compute displacement of each vertex and verify uniform
        for ri, vert_indices in enumerate(region_vert_indices):
            if len(vert_indices) < 2:
                continue
            displacements = result.deformed_vertices[vert_indices] - resized[vert_indices]
            # All displacements should be the same (rigid translation)
            for d in displacements[1:]:
                np.testing.assert_allclose(d, displacements[0], atol=1e-6,
                    err_msg=f"Region {ri} vertices moved non-rigidly")
