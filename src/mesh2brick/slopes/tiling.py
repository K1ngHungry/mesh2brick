"""Place slope bricks in a voxel grid based on detected slope regions."""

import numpy as np
import open3d as o3d

from mesh2brick.data.brick_structure import Brick, SlopeBrick, _SLOPE_DIR_TO_ROTATION
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


def _merge_slopes(
    bricks: list[Brick],
    slope_direction: int,
    matched_bricks: list[dict],
    n_lateral: int,
) -> list[Brick]:
    """Merge adjacent small slope bricks into larger ones along the lateral axis.

    Prioritises corners then edges: bricks are sorted so that the lateral
    extremes (first/last positions) are merged first, giving them the best
    chance of getting a wide brick that connects to a neighbour.
    """
    if not bricks:
        return bricks

    base = bricks[0]
    available_widths = sorted(
        set(b['width'] for b in matched_bricks
            if b['length'] == base.l and b['height'] == base.h),
        reverse=True,
    )
    if len(available_widths) <= 1:
        return bricks

    lateral_is_y = slope_direction in (0, 2)
    base_step = base.w  # lateral footprint of the smallest brick

    # Group by (slope_axis_pos, z)
    groups: dict[tuple[int, int], list[Brick]] = {}
    for b in bricks:
        key = (b.x, b.z) if lateral_is_y else (b.y, b.z)
        groups.setdefault(key, []).append(b)

    merged: list[Brick] = []
    for group in groups.values():
        group.sort(key=lambda b: b.y if lateral_is_y else b.x)
        n = len(group)

        # Build priority: corners (first/last) highest, then edges near them
        priority = [0.0] * n
        for idx in range(n):
            dist_from_edge = min(idx, n - 1 - idx)
            priority[idx] = dist_from_edge  # lower = higher priority

        # Greedy merge in priority order (process lowest-priority-value first)
        used = [False] * n
        pending: list[tuple[float, int]] = sorted(
            ((priority[idx], idx) for idx in range(n)),
        )

        for _, start_idx in pending:
            if used[start_idx]:
                continue
            placed = False
            for w in available_widths:
                count = w // base_step
                # Try to extend from start_idx using contiguous unused bricks
                run = [start_idx]
                # Extend forward
                j = start_idx + 1
                while len(run) < count and j < n and not used[j]:
                    lat_pos = (group[j].y if lateral_is_y else group[j].x)
                    expected = (group[run[0]].y if lateral_is_y else group[run[0]].x) + len(run) * base_step
                    if lat_pos == expected:
                        run.append(j)
                        j += 1
                    else:
                        break
                # Extend backward if needed
                j = start_idx - 1
                while len(run) < count and j >= 0 and not used[j]:
                    lat_pos = (group[j].y if lateral_is_y else group[j].x)
                    expected = (group[run[0]].y if lateral_is_y else group[run[0]].x) - base_step
                    if lat_pos == expected:
                        run.insert(0, j)
                        j -= 1
                    else:
                        break
                if len(run) == count:
                    b0 = group[run[0]]
                    merged.append(SlopeBrick(
                        l=base.l, w=w, h=base.h,
                        slope_direction=slope_direction, x=b0.x, y=b0.y, z=b0.z))
                    for idx in run:
                        used[idx] = True
                    placed = True
                    break
            if not placed:
                used[start_idx] = True
                merged.append(group[start_idx])
    return merged


def _tile_candidates(
    slope_direction: int,
    x_min: int, y_min: int, z_min: int,
    n_x: int, n_y: int, n_z: int,
    brick_l: int, brick_w: int, brick_h: int,
    dim_x: int, dim_y: int,
    step_x: int, step_y: int,
    world_shape: tuple[int, ...]
) -> list[SlopeBrick]:
    """Place a staircase of slope bricks for one region."""
    placed = []
    for step_z in range(n_z):
        z = z_min + step_z * brick_h
        if z + brick_h > world_shape[2]:
            continue

        if slope_direction == 0:  # +X
            start_x = x_min + (n_z - 1 - step_z) * step_x
            n_step_x = 1
            n_step_y = n_y
        elif slope_direction == 2:  # -X
            start_x = x_min + step_z * step_x
            n_step_x = 1
            n_step_y = n_y
        elif slope_direction == 1:  # +Y
            start_y = y_min + (n_z - 1 - step_z) * step_y
            n_step_y = 1
            n_step_x = n_x
        elif slope_direction == 3:  # -Y
            start_y = y_min + step_z * step_y
            n_step_y = 1
            n_step_x = n_x

        for ix in range(max(1, n_step_x)):
            for iy in range(max(1, n_step_y)):
                x = (start_x if slope_direction in (0, 2) else x_min + ix * dim_x)
                y = (start_y if slope_direction in (1, 3) else y_min + iy * dim_y)

                if x >= 0 and y >= 0 and x + dim_x <= world_shape[0] and y + dim_y <= world_shape[1]:
                    placed.append(SlopeBrick(l=brick_l, w=brick_w, h=brick_h,
                                             slope_direction=slope_direction, x=int(x), y=int(y), z=int(z)))
    return placed


def place_slope_bricks(
    voxels: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    voxel_origin: np.ndarray | None = None,
    region_n_steps: list[int] | None = None,
    verbose: bool = False,
) -> tuple[list[SlopeBrick], np.ndarray]:
    """Place slope bricks and return (slope_bricks, remaining_voxels).

    For each slope region, generates a staircase of candidate bricks via
    _tile_candidates, then validates each candidate top-down:

    1. Bounds check — brick must fit within the voxel grid.
    2. Overlap check — brick must not overlap already-placed slope bricks.
    3. Support check — if the brick's footprint has no remaining voxels,
       it is still accepted at ridges (where two slopes meet) if voxels
       exist in the layer directly below. Otherwise rejected.
    4. Noise check — stray voxels in the air triangle (h=3) or above the
       bbox (h=2) are tolerated up to a threshold, then cleared.
    5. Above check — if more than one voxel exists above the slope column
       in the original grid, the brick is rejected (indicates solid geometry
       above, not a slope surface).

    Accepted bricks have their voxels cleared from `remaining`, then are
    merged into wider bricks where possible.
    """
    remaining = voxels.astype(bool).copy()
    slope_bricks: list[SlopeBrick] = []
    world_shape = voxels.shape

    if voxel_origin is None:
        voxel_origin = np.asarray(mesh.get_min_bound())

    for region_idx, (region, matched_bricks) in enumerate(assignments):
        if not matched_bricks:
            continue

        best_brick_info = min(matched_bricks, key=lambda b: b['length'] * b['width'])

        brick_l = best_brick_info['length']
        brick_w = best_brick_info['width']
        brick_h = best_brick_info['height']
        rotation = _SLOPE_DIR_TO_ROTATION[region.slope_direction]
        dim_x, dim_y = SlopeBrick.rotated_dim(brick_l, brick_w, rotation)

        run = brick_l - 1 if brick_h == 3 and brick_l > 1 else brick_l
        step_x, step_y = SlopeBrick.rotated_dim(run, brick_w, rotation)

        x_min, x_max, y_min, y_max, z_min, z_max = _region_voxel_bounds(
            mesh, region, voxel_origin)

        n_x = max(1, round((x_max - x_min + 1) / step_x))
        n_y = max(1, round((y_max - y_min + 1) / step_y))

        if region_n_steps is not None:
            n_z = region_n_steps[region_idx]
        else:
            n_z = max(1, round((z_max - z_min + 1) / brick_h))

        if region.slope_direction in (0, 2):
            n_z = min(n_z, n_x)
        elif region.slope_direction in (1, 3):
            n_z = min(n_z, n_y)

        placed = _tile_candidates(
            region.slope_direction,
            x_min, y_min, z_min,
            n_x, n_y, n_z,
            brick_l, brick_w, brick_h, dim_x, dim_y,
            step_x, step_y, world_shape
        )

        placed.sort(key=lambda b: -b.z)
        accepted = []
        rejected_bounds = rejected_no_voxels = rejected_noise = rejected_above = 0
        for brick in placed:
            sx, sy, sz = brick.slice

            # Bounds check
            if sx.stop > world_shape[0] or sy.stop > world_shape[1] or sz.stop > world_shape[2]:
                rejected_bounds += 1
                continue

            # Overlap check
            if any(brick.overlaps(existing) for existing in slope_bricks if existing.type == 1):
                rejected_no_voxels += 1
                continue

            # Support check
            if not remaining[brick.slice].any():
                sx, sy, sz = brick.slice
                if brick.z > 0 and voxels[sx, sy, brick.z - 1].any():
                    pass  # Ridge — supported from below
                else:
                    rejected_no_voxels += 1
                    continue

            # Noise check
            slope_sl, stud_sl = brick.slope_slice()
            slope_sx, slope_sy, slope_sz = slope_sl

            if stud_sl is not None:
                air_start = brick.z + 1
                air_end = min(brick.z + brick_h, world_shape[2])
                noise_count = int(remaining[slope_sx, slope_sy, air_start:air_end].sum()) if air_start < air_end else 0
            else:
                z_above = brick.z + brick_h
                noise_count = int(remaining[slope_sx, slope_sy, z_above:z_above + 1].sum()) if z_above < world_shape[2] else 0

            max_noise = (brick_h - 1) if stud_sl is not None else 1
            if noise_count > max_noise:
                rejected_noise += 1
                continue

            # Above check (uses original voxel grid, not remaining)
            z_above = brick.z + brick_h
            above_count = int(voxels[slope_sx, slope_sy, z_above:].sum()) if z_above < world_shape[2] else 0
            if above_count > 1:
                rejected_above += 1
                continue

            # Accept: clear voxels
            remaining[brick.slice] = False
            if 0 < noise_count <= max_noise:
                if stud_sl is not None:
                    remaining[slope_sx, slope_sy, air_start:air_end] = False
                else:
                    remaining[slope_sx, slope_sy, z_above:z_above + 1] = False
            if above_count == 1:
                remaining[slope_sx, slope_sy, z_above:] = False

            accepted.append(brick)

        n_lateral = n_y if region.slope_direction in (0, 2) else n_x
        final = _merge_slopes(accepted, region.slope_direction, matched_bricks, n_lateral)
        slope_bricks.extend(final)

        if verbose and placed:
            print(f"    Region {region_idx}: {len(final)}/{len(placed)} placed "
                  f"(rejected: bounds={rejected_bounds}, no_voxels={rejected_no_voxels}, "
                  f"noise={rejected_noise}, above={rejected_above})")

    return slope_bricks, remaining
