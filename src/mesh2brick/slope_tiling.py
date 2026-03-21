"""Place slope bricks in a voxel grid based on detected slope regions."""

import numpy as np
import open3d as o3d

from mesh2brick.data.brick_library import brick_library
from mesh2brick.data.brick_structure import Brick
from mesh2brick.slope_detection import SlopeRegion


def _region_voxel_bounds(
    mesh: o3d.geometry.TriangleMesh,
    region: SlopeRegion,
    voxel_origin: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Compute the integer voxel-space bounding box of a slope region.

    Converts mesh-space vertex positions to voxel grid indices by subtracting
    the voxel grid origin.

    Returns (x_min, x_max, y_min, y_max, z_min, z_max) where max is exclusive.
    """
    triangles = np.asarray(mesh.triangles)[region.face_indices]
    vertex_indices = np.unique(triangles)
    vertices = np.asarray(mesh.vertices)[vertex_indices]

    # Convert from mesh world coordinates to voxel grid indices
    voxel_coords = vertices - voxel_origin

    mins = np.floor(voxel_coords.min(axis=0)).astype(int)
    maxs = np.ceil(voxel_coords.max(axis=0)).astype(int)
    return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]


def _slope_direction_to_rotation(slope_direction: int) -> int:
    """Map slope_direction (0=+X, 1=+Y, 2=-X, 3=-Y) to brick rotation.

    Based on LDR matrix analysis (applying rotation to direction (0,0,-1)):
      R0 (identity) → thin end faces -Y in our coords (LDR -Z)
      R1 (90°)      → thin end faces +X in our coords (LDR +X)
      R2 (180°)     → thin end faces +Y in our coords (LDR +Z)
      R3 (270°)     → thin end faces -X in our coords (LDR -X)
    """
    # slope_direction: 0=+X → R1, 1=+Y → R2, 2=-X → R3, 3=-Y → R0
    mapping = {0: 1, 1: 2, 2: 3, 3: 0}
    return mapping[slope_direction]


def _physical_footprint(brick_l: int, brick_w: int, rotation: int) -> tuple[int, int]:
    """Get (x_extent, y_extent) for slope bricks accounting for rotation.

    For slope bricks, l (slope run) maps to Y for even rotations and X for odd.
    This is opposite of regular bricks.
    """
    if rotation % 2 == 0:
        return brick_w, brick_l  # even: x=w, y=l
    return brick_l, brick_w  # odd: x=l, y=w


def place_slope_bricks(
    voxels: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    voxel_origin: np.ndarray | None = None,
) -> tuple[list[Brick], np.ndarray]:
    """Place slope bricks in the voxel grid for each assigned slope region.

    Args:
        voxels: 3D bool/uint8 array of occupied voxels (after Z-scaling).
        mesh: Deformed + Z-scaled Open3D mesh (vertices at voxel scale).
        assignments: List of (SlopeRegion, matched_bricks) from compute_optimal_scale.
        voxel_origin: Origin of the voxel grid (mesh min bound). If None, uses mesh min bound.

    Returns:
        slope_bricks: List of placed Brick objects (type=1).
        remaining_voxels: Copy of voxels with slope-brick voxels zeroed out.
    """
    remaining = voxels.astype(bool).copy()
    slope_bricks: list[Brick] = []
    world_shape = voxels.shape

    if voxel_origin is None:
        voxel_origin = np.asarray(mesh.get_min_bound())

    for region, matched_bricks in assignments:
        if not matched_bricks:
            continue

        x_min, x_max, y_min, y_max, z_min, z_max = _region_voxel_bounds(
            mesh, region, voxel_origin)

        # Clamp to world bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(world_shape[0], x_max)
        y_max = min(world_shape[1], y_max)
        z_max = min(world_shape[2], z_max)

        region_dx = x_max - x_min
        region_dy = y_max - y_min
        region_dz = z_max - z_min

        if region_dx <= 0 or region_dy <= 0 or region_dz <= 0:
            continue

        rotation = _slope_direction_to_rotation(region.slope_direction)

        # Select the best brick that fits in the region
        best_brick_info = _select_brick(matched_bricks, region_dx, region_dy, region_dz, rotation)
        if best_brick_info is None:
            continue

        brick_l = best_brick_info['length']
        brick_w = best_brick_info['width']
        brick_h = best_brick_info['height']
        foot_x, foot_y = _physical_footprint(brick_l, brick_w, rotation)

        # Tile the slope surface with bricks
        placed = _tile_slope_surface(
            voxels, region.slope_direction,
            x_min, x_max, y_min, y_max, z_min, z_max,
            brick_l, brick_w, brick_h, foot_x, foot_y, rotation, world_shape,
        )
        # Clear voxels occupied by slope bricks, plus a buffer above each brick
        # to prevent regular bricks from visually clipping through the wedge.
        for brick in placed:
            remaining[brick.slice] = False
            # Clear voxels above the slope brick (the "air" above the wedge)
            sx, sy, _ = brick.slice
            z_above_start = brick.z + brick_h
            z_above_end = min(z_above_start + brick_h, world_shape[2])
            if z_above_start < z_above_end:
                remaining[sx, sy, z_above_start:z_above_end] = False
        slope_bricks.extend(placed)

    return slope_bricks, remaining


def _select_brick(
    matched_bricks: list[dict],
    region_dx: int,
    region_dy: int,
    region_dz: int,
    rotation: int,
) -> dict | None:
    """Select the largest matching brick that fits within the region bounds."""
    for brick_info in matched_bricks:  # Already sorted largest-first
        foot_x, foot_y = _physical_footprint(brick_info['length'], brick_info['width'], rotation)
        if foot_x <= region_dx and foot_y <= region_dy and brick_info['height'] <= region_dz:
            return brick_info
    return None


def _tile_slope_surface(
    remaining: np.ndarray,
    slope_direction: int,
    x_min: int, x_max: int,
    y_min: int, y_max: int,
    z_min: int, z_max: int,
    brick_l: int, brick_w: int, brick_h: int,
    foot_x: int, foot_y: int,
    rotation: int,
    world_shape: tuple[int, ...],
) -> list[Brick]:
    """Tile the slope surface with bricks following the diagonal.

    Instead of filling the entire bounding box, places bricks along the slope
    surface. For each step up in Z (by brick_h), we step along the slope
    direction (by brick_l) to follow the diagonal.
    """
    placed = []

    # The slope runs along one axis and rises along Z.
    # slope_direction: 0=+X, 1=+Y, 2=-X, 3=-Y
    # For +X: surface is at high X at bottom, low X at top → start at x_max, step -foot_x per z-step
    # For -X: surface is at low X at bottom, high X at top → start at x_min, step +foot_x per z-step
    # For +Y: surface is at high Y at bottom, low Y at top → start at y_max, step -foot_y per z-step
    # For -Y: surface is at low Y at bottom, high Y at top → start at y_min, step +foot_y per z-step

    n_z_steps = (z_max - z_min) // brick_h

    for step in range(n_z_steps):
        z = z_min + step * brick_h
        if z + brick_h > world_shape[2]:
            break

        # Compute the position along the slope direction for this Z level
        if slope_direction == 0:  # +X: start at right edge, move left as we go up
            x_pos = x_max - (step + 1) * foot_x
            for y in range(y_min, y_max, foot_y):
                if y + foot_y > world_shape[1]:
                    break
                brick = _try_place(remaining, brick_l, brick_w, brick_h, rotation,
                                   x_pos, y, z, world_shape)
                if brick:
                    placed.append(brick)

        elif slope_direction == 2:  # -X: start at left edge, move right as we go up
            x_pos = x_min + step * foot_x
            for y in range(y_min, y_max, foot_y):
                if y + foot_y > world_shape[1]:
                    break
                brick = _try_place(remaining, brick_l, brick_w, brick_h, rotation,
                                   x_pos, y, z, world_shape)
                if brick:
                    placed.append(brick)

        elif slope_direction == 1:  # +Y: start at far edge, move near as we go up
            y_pos = y_max - (step + 1) * foot_y
            for x in range(x_min, x_max, foot_x):
                if x + foot_x > world_shape[0]:
                    break
                brick = _try_place(remaining, brick_l, brick_w, brick_h, rotation,
                                   x, y_pos, z, world_shape)
                if brick:
                    placed.append(brick)

        elif slope_direction == 3:  # -Y: start at near edge, move far as we go up
            y_pos = y_min + step * foot_y
            for x in range(x_min, x_max, foot_x):
                if x + foot_x > world_shape[0]:
                    break
                brick = _try_place(remaining, brick_l, brick_w, brick_h, rotation,
                                   x, y_pos, z, world_shape)
                if brick:
                    placed.append(brick)

    return placed


def _try_place(
    voxels: np.ndarray,
    brick_l: int, brick_w: int, brick_h: int,
    rotation: int,
    x: int, y: int, z: int,
    world_shape: tuple[int, ...],
) -> Brick | None:
    """Try to place a slope brick at the given position. Returns the Brick if placed, None otherwise."""
    if x < 0 or y < 0 or z < 0:
        return None

    brick = Brick(type=1, l=brick_l, w=brick_w, h=brick_h, rotation=rotation, x=x, y=y, z=z)

    # Check bounds
    sx, sy, sz = brick.slice
    if sx.stop > world_shape[0] or sy.stop > world_shape[1] or sz.stop > world_shape[2]:
        return None

    # Check that the brick volume has enough filled voxels
    vol = voxels[brick.slice]
    if vol.size == 0:
        return None
    fill_ratio = vol.sum() / vol.size
    if fill_ratio < 0.3:
        return None

    return brick
