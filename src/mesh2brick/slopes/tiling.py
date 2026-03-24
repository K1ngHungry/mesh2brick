"""Place slope bricks in a voxel grid based on detected slope regions."""

import numpy as np
import open3d as o3d

from mesh2brick.data.brick_structure import Brick
from .detection import SlopeRegion


def _region_voxel_bounds(
    mesh: o3d.geometry.TriangleMesh,
    region: SlopeRegion,
    voxel_origin: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Compute the integer voxel-space bounding box of a slope region."""
    triangles = np.asarray(mesh.triangles)[region.face_indices]
    vertex_indices = np.unique(triangles)
    vertices = np.asarray(mesh.vertices)[vertex_indices]

    voxel_coords = vertices - voxel_origin

    mins = np.floor(voxel_coords.min(axis=0)).astype(int)
    maxs = np.round(voxel_coords.max(axis=0)).astype(int)
    return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]


def _slope_direction_to_rotation(slope_direction: int) -> int:
    """Map slope_direction (0=+X, 1=+Y, 2=-X, 3=-Y) to brick rotation."""
    mapping = {0: 1, 1: 2, 2: 3, 3: 0}
    return mapping[slope_direction]


def _physical_footprint(brick_l: int, brick_w: int, rotation: int) -> tuple[int, int]:
    if rotation % 2 == 0:
        return brick_w, brick_l
    return brick_l, brick_w


def _tile_rigid_block(
    slope_direction: int,
    x_min: int, y_min: int, z_min: int,
    n_x: int, n_y: int, n_z: int,
    brick_l: int, brick_w: int, brick_h: int,
    foot_x: int, foot_y: int,
    rotation: int,
    world_shape: tuple[int, ...]
) -> list[Brick]:
    """Place a staircase of slope bricks for one region."""
    placed = []
    for step_z in range(n_z):
        z = z_min + step_z * brick_h
        if z + brick_h > world_shape[2]:
            continue

        if slope_direction == 0:  # +X
            start_x = x_min + (n_z - 1 - step_z) * foot_x
            n_step_x = 1
            n_step_y = n_y
        elif slope_direction == 2:  # -X
            start_x = x_min + step_z * foot_x
            n_step_x = 1
            n_step_y = n_y
        elif slope_direction == 1:  # +Y
            start_y = y_min + (n_z - 1 - step_z) * foot_y
            n_step_y = 1
            n_step_x = n_x
        elif slope_direction == 3:  # -Y
            start_y = y_min + step_z * foot_y
            n_step_y = 1
            n_step_x = n_x

        for ix in range(max(1, n_step_x)):
            for iy in range(max(1, n_step_y)):
                x = (start_x if slope_direction in (0, 2) else x_min + ix * foot_x)
                y = (start_y if slope_direction in (1, 3) else y_min + iy * foot_y)

                if x >= 0 and y >= 0 and x + foot_x <= world_shape[0] and y + foot_y <= world_shape[1]:
                    placed.append(Brick(type=1, l=brick_l, w=brick_w, h=brick_h,
                                        rotation=rotation, x=int(x), y=int(y), z=int(z)))
    return placed


def place_slope_bricks(
    voxels: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    voxel_origin: np.ndarray | None = None,
) -> tuple[list[Brick], np.ndarray]:
    """Place slope bricks and return (slope_bricks, remaining_voxels)."""
    remaining = voxels.astype(bool).copy()
    slope_bricks: list[Brick] = []
    world_shape = voxels.shape

    if voxel_origin is None:
        voxel_origin = np.asarray(mesh.get_min_bound())

    for region, matched_bricks in assignments:
        if not matched_bricks:
            continue

        best_brick_info = min(matched_bricks, key=lambda b: b['length'] * b['width'])

        brick_l = best_brick_info['length']
        brick_w = best_brick_info['width']
        brick_h = best_brick_info['height']
        rotation = _slope_direction_to_rotation(region.slope_direction)
        foot_x, foot_y = _physical_footprint(brick_l, brick_w, rotation)

        x_min, x_max, y_min, y_max, z_min, z_max = _region_voxel_bounds(
            mesh, region, voxel_origin)

        n_x = max(1, round((x_max - x_min + 1) / foot_x))
        n_y = max(1, round((y_max - y_min + 1) / foot_y))
        n_z = max(1, round((z_max - z_min + 1) / brick_h))

        # Cap n_z so the staircase doesn't exceed the horizontal extent
        if region.slope_direction in (0, 2):
            n_z = min(n_z, n_x)
        elif region.slope_direction in (1, 3):
            n_z = min(n_z, n_y)

        placed = _tile_rigid_block(
            region.slope_direction,
            x_min, y_min, z_min,
            n_x, n_y, n_z,
            brick_l, brick_w, brick_h, foot_x, foot_y, rotation, world_shape
        )

        for brick in placed:
            sx, sy, sz = brick.slice
            if sx.stop <= world_shape[0] and sy.stop <= world_shape[1] and sz.stop <= world_shape[2]:
                remaining[brick.slice] = False
                # Clear voxels above — the staircase defines the outer surface
                z_above_start = brick.z + brick_h
                if z_above_start < world_shape[2]:
                    remaining[sx, sy, z_above_start:] = False
        slope_bricks.extend(placed)

    return slope_bricks, remaining
