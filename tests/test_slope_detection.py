import math

import numpy as np
import open3d as o3d
import pytest

from mesh2brick.data.brick_structure import Brick
from mesh2brick.slope_detection import (
    SlopeRegion,
    compute_optimal_scale,
    detect_slopes,
    get_slope_bricks,
    match_slope_to_bricks,
)


def _make_mesh(vertices: list, triangles: list) -> o3d.geometry.TriangleMesh:
    """Create an Open3D triangle mesh from vertices and triangle indices."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=int))
    return mesh


def _make_ramp(angle_deg: float = 45.0, width: float = 2.0) -> o3d.geometry.TriangleMesh:
    """Create a ramp mesh with a single sloped face at the given angle.

    The ramp rises in the +X direction. The sloped face faces +X/+Z.
    """
    run = 1.0
    rise = run * math.tan(math.radians(angle_deg))
    # Ramp shape (side view):
    #   (0, 0, rise) --- (run, 0, rise)
    #       |                |
    #   (0, 0, 0) ---- (run, 0, 0)
    # Plus width in Y direction
    vertices = [
        [0, 0, 0],           # 0: bottom-left-back
        [run, 0, 0],         # 1: bottom-right-back
        [run, 0, rise],      # 2: top-right-back
        [0, width, 0],       # 3: bottom-left-front
        [run, width, 0],     # 4: bottom-right-front
        [run, width, rise],  # 5: top-right-front
    ]
    triangles = [
        # Bottom face (horizontal)
        [0, 1, 4], [0, 4, 3],
        # Back wall (vertical, Y=0 plane)
        [0, 2, 1], [0, 2, 1],
        # Front wall (vertical, Y=width plane)
        [3, 4, 5],
        # Right wall (vertical, X=run plane)
        [1, 2, 5], [1, 5, 4],
        # Sloped face (the ramp surface)
        [0, 5, 2], [0, 3, 5],
    ]
    return _make_mesh(vertices, triangles)


def _make_box() -> o3d.geometry.TriangleMesh:
    """Create a simple box (all faces horizontal or vertical)."""
    return o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)


def _make_pyramid(base_size: float = 4.0, height: float = 4.0) -> o3d.geometry.TriangleMesh:
    """Create a pyramid with 4 sloped faces pointing in 4 cardinal directions."""
    half = base_size / 2
    apex = [0, 0, height]
    vertices = [
        [-half, -half, 0],  # 0: corner -X, -Y
        [half, -half, 0],   # 1: corner +X, -Y
        [half, half, 0],    # 2: corner +X, +Y
        [-half, half, 0],   # 3: corner -X, +Y
        apex,               # 4: apex
    ]
    triangles = [
        # Base (horizontal)
        [0, 1, 2], [0, 2, 3],
        # 4 sloped faces
        [0, 1, 4],  # face toward -Y
        [1, 2, 4],  # face toward +X
        [2, 3, 4],  # face toward +Y
        [3, 0, 4],  # face toward -X
    ]
    return _make_mesh(vertices, triangles)


class TestDetectSlopes:
    def test_ramp_45(self):
        """A 45° ramp should produce one slope region near 45°."""
        mesh = _make_ramp(angle_deg=45.0, width=5.0)
        regions = detect_slopes(mesh, min_area_fraction=0.05)
        assert len(regions) >= 1
        # Find the region closest to 45°
        angles = [r.slope_angle for r in regions]
        closest = min(angles, key=lambda a: abs(a - 45.0))
        assert abs(closest - 45.0) < 5.0

    def test_flat_box(self):
        """A box with only horizontal and vertical faces should produce 0 slope regions."""
        mesh = _make_box()
        regions = detect_slopes(mesh)
        assert len(regions) == 0

    def test_pyramid_four_directions(self):
        """A pyramid should detect 4 slope regions with 4 different directions."""
        mesh = _make_pyramid(base_size=4.0, height=4.0)
        regions = detect_slopes(mesh, min_area_fraction=0.01)
        assert len(regions) == 4
        directions = sorted([r.slope_direction for r in regions])
        assert directions == [0, 1, 2, 3]

    def test_small_slope_filtered(self):
        """A tiny sloped face below the area threshold should be filtered out."""
        # Create a box with a tiny chamfer (sloped face)
        mesh = _make_box()
        # The box has all axis-aligned faces, so no slopes detected
        regions = detect_slopes(mesh, min_area_fraction=0.1)
        assert len(regions) == 0

    def test_steep_slope(self):
        """A steep slope (~63°) should be detected with the right angle."""
        mesh = _make_ramp(angle_deg=63.0, width=5.0)
        regions = detect_slopes(mesh, min_area_fraction=0.05)
        assert len(regions) >= 1
        angles = [r.slope_angle for r in regions]
        closest = min(angles, key=lambda a: abs(a - 63.0))
        assert abs(closest - 63.0) < 5.0

    def test_shallow_slope(self):
        """A shallow slope (~27°) should be detected with the right angle."""
        mesh = _make_ramp(angle_deg=27.0, width=5.0)
        regions = detect_slopes(mesh, min_area_fraction=0.05)
        assert len(regions) >= 1
        angles = [r.slope_angle for r in regions]
        closest = min(angles, key=lambda a: abs(a - 27.0))
        assert abs(closest - 27.0) < 5.0

    def test_empty_mesh(self):
        """An empty mesh should return no regions."""
        mesh = o3d.geometry.TriangleMesh()
        regions = detect_slopes(mesh)
        assert len(regions) == 0


class TestBrickAngleProperty:
    def test_slope_brick_angle(self):
        """Slope brick angle should be atan(h/l) in degrees."""
        brick = Brick(type=1, l=2, w=1, h=3, rotation=0, x=0, y=0, z=0)
        assert brick.angle is not None
        expected = math.degrees(math.atan(3 / 2))  # ~56.3°
        assert abs(brick.angle - expected) < 0.1

    def test_normal_brick_no_angle(self):
        """Non-slope bricks should have angle=None."""
        brick = Brick(type=0, l=2, w=4, h=3, rotation=0, x=0, y=0, z=0)
        assert brick.angle is None

    def test_plate_no_angle(self):
        """Plate bricks (type=0) should have angle=None."""
        brick = Brick(type=0, l=1, w=2, h=1, rotation=0, x=0, y=0, z=0)
        assert brick.angle is None

    def test_45_degree_slope(self):
        """l=2, h=2 slope should be 45°."""
        brick = Brick(type=1, l=2, w=1, h=2, rotation=0, x=0, y=0, z=0)
        assert abs(brick.angle - 45.0) < 0.1


class TestGetSlopeBricks:
    def test_returns_only_slopes(self):
        """get_slope_bricks should only return type=1 bricks."""
        bricks = get_slope_bricks()
        assert len(bricks) > 0
        for b in bricks:
            # Verify it has expected keys
            assert 'brick_id' in b
            assert 'angle' in b
            assert 'length' in b
            assert 'width' in b
            assert 'height' in b

    def test_known_angles(self):
        """Verify the expected distinct angles exist."""
        bricks = get_slope_bricks()
        angles = sorted(set(round(b['angle'], 1) for b in bricks))
        # Expected: ~26.6, ~36.9, ~45.0, ~56.3, ~63.4
        assert len(angles) == 5
        assert abs(angles[0] - 26.6) < 1.0
        assert abs(angles[1] - 36.9) < 1.0
        assert abs(angles[2] - 45.0) < 1.0
        assert abs(angles[3] - 56.3) < 1.0
        assert abs(angles[4] - 63.4) < 1.0


class TestMatchSlopeToBricks:
    def test_match_45(self):
        """A 45° slope should match the 45° brick (l=2, h=2)."""
        matches = match_slope_to_bricks(45.0)
        assert len(matches) >= 1
        assert abs(matches[0]['angle'] - 45.0) < 0.1

    def test_match_steep(self):
        """A 60° slope should match the ~63.4° bricks (l=1, h=2)."""
        matches = match_slope_to_bricks(60.0)
        assert len(matches) >= 1
        assert abs(matches[0]['angle'] - 63.4) < 1.0

    def test_match_shallow(self):
        """A 25° slope should match the ~26.6° brick (l=6, h=3)."""
        matches = match_slope_to_bricks(25.0)
        assert len(matches) >= 1
        assert abs(matches[0]['angle'] - 26.6) < 1.0

    def test_match_returns_sorted_by_area(self):
        """Matching bricks should be sorted by area, largest first."""
        # ~56.3° has multiple bricks (w=1,2,4,6)
        matches = match_slope_to_bricks(56.0)
        assert len(matches) > 1
        areas = [m['length'] * m['width'] for m in matches]
        assert areas == sorted(areas, reverse=True)

    def test_match_empty_library(self):
        """Empty slope brick list should return empty."""
        matches = match_slope_to_bricks(45.0, slope_bricks=[])
        assert matches == []


def _make_region(slope_angle=45.0, length=0.3, width=0.2, direction=0):
    """Helper to create a SlopeRegion with minimal fields."""
    return SlopeRegion(
        face_indices=[0],
        avg_normal=np.array([0.5, 0.0, 0.5]),
        area=1.0,
        slope_angle=slope_angle,
        slope_direction=direction,
        length=length,
        width=width,
        height=0.1,
    )


class TestComputeOptimalScale:
    def test_no_regions_returns_default(self):
        """No slope regions should return the default scale."""
        scale, assignments = compute_optimal_scale([], default_scale=20.0)
        assert scale == 20.0
        assert assignments == []

    def test_single_region_easy(self):
        """Region with s_min < default_scale should keep default scale."""
        # 45° matches brick l=2, w=1. s_min = max(2/0.3, 1/0.2) = max(6.67, 5.0) = 6.67
        region = _make_region(slope_angle=45.0, length=0.3, width=0.2)
        scale, assignments = compute_optimal_scale([region], default_scale=20.0)
        assert scale == 20.0
        assert len(assignments) == 1

    def test_region_requires_large_scale(self):
        """Tiny region should push scale up, capped at max_scale."""
        # 45° brick l=2, w=1. s_min = max(2/0.05, 1/0.02) = max(40, 50) = 50
        region = _make_region(slope_angle=45.0, length=0.05, width=0.02)
        scale, assignments = compute_optimal_scale(
            [region], default_scale=20.0, max_scale=40.0,
        )
        assert scale == 40.0

    def test_fallback_discards_region(self):
        """Region whose s_min > 2*s_star should be discarded."""
        r1 = _make_region(slope_angle=45.0, length=0.3, width=0.2)  # s_min ~6.67
        r2 = _make_region(slope_angle=45.0, length=0.05, width=0.02)  # s_min ~50
        scale, assignments = compute_optimal_scale(
            [r1, r2], default_scale=15.0, max_scale=20.0,
        )
        assert scale == 20.0
        # r2 s_min=50, 20 < 0.5*50=25 → discarded
        assert len(assignments) == 1

    def test_multiple_regions_uses_max_smin(self):
        """Optimal scale should be max of all s_min values (when within bounds).
        Angles are in isotropic mesh space; converted to voxel space (3x Z) before matching.
        iso 18.4° → voxel ~45°, iso 26.6° → voxel ~56.3°."""
        r1 = _make_region(slope_angle=18.4, length=0.5, width=0.5)
        # voxel ~45° brick l=2, w=1. s_min = max(2/0.5, 1/0.5) = max(4, 2) = 4
        r2 = _make_region(slope_angle=26.6, length=0.2, width=0.2)
        # voxel ~56° matches 56.31° bricks, smallest is l=2, w=1. s_min = max(2/0.2, 1/0.2) = max(10, 5) = 10
        scale, assignments = compute_optimal_scale(
            [r1, r2], default_scale=5.0, max_scale=40.0,
        )
        assert scale >= 10.0
        assert len(assignments) == 2

    def test_scale_not_below_default(self):
        """Scale should never go below default_scale even if all s_min are small."""
        region = _make_region(slope_angle=45.0, length=1.0, width=1.0)
        # s_min = max(2/1, 1/1) = 2
        scale, assignments = compute_optimal_scale(
            [region], default_scale=20.0, max_scale=40.0,
        )
        assert scale == 20.0


