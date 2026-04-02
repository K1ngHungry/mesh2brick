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


def _slope_region_slice(
    brick: Brick, slope_direction: int,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice] | None]:
    """Return (slope_slice, stud_slice) for the sloping vs stud portions.

    For h=3 l=2 bricks: one column is stud (full height), one is slope.
    For h=2 or l=1 bricks: entire footprint is slope, stud_slice is None.
    """
    sx, sy, sz = brick.slice
    run = brick.l - 1 if brick.h == 3 and brick.l > 1 else brick.l
    if run == brick.l:
        return (sx, sy, sz), None

    # h=3, l=2: one column is stud, one is slope
    # The stud is at the HIGH end of the slope direction (where the slope rises to)
    if slope_direction in (0, 2):
        # X-axis slope: split along x
        if slope_direction == 0:  # +X: slope rises toward +X, stud at high x
            slope_x = slice(sx.start, sx.start + 1)
            stud_x = slice(sx.stop - 1, sx.stop)
        else:  # -X: slope rises toward -X, stud at low x
            slope_x = slice(sx.stop - 1, sx.stop)
            stud_x = slice(sx.start, sx.start + 1)
        return (slope_x, sy, sz), (stud_x, sy, sz)
    else:
        # Y-axis slope: split along y
        if slope_direction == 1:  # +Y: stud at high y
            slope_y = slice(sy.start, sy.start + 1)
            stud_y = slice(sy.stop - 1, sy.stop)
        else:  # -Y: stud at low y
            slope_y = slice(sy.stop - 1, sy.stop)
            stud_y = slice(sy.start, sy.start + 1)
        return (sx, slope_y, sz), (sx, stud_y, sz)


def _merge_slope_bricks(
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
                    merged.append(Brick(
                        type=1, l=base.l, w=w, h=base.h,
                        rotation=b0.rotation, x=b0.x, y=b0.y, z=b0.z))
                    for idx in run:
                        used[idx] = True
                    placed = True
                    break
            if not placed:
                used[start_idx] = True
                merged.append(group[start_idx])
    return merged


def _tile_rigid_block(
    slope_direction: int,
    x_min: int, y_min: int, z_min: int,
    n_x: int, n_y: int, n_z: int,
    brick_l: int, brick_w: int, brick_h: int,
    foot_x: int, foot_y: int,
    step_x: int, step_y: int,
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
    region_n_steps: list[int] | None = None,
) -> tuple[list[Brick], np.ndarray]:
    """Place slope bricks and return (slope_bricks, remaining_voxels)."""
    remaining = voxels.astype(bool).copy()
    slope_bricks: list[Brick] = []
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
        rotation = _slope_direction_to_rotation(region.slope_direction)
        foot_x, foot_y = _physical_footprint(brick_l, brick_w, rotation)

        # Effective run for staircase step advance (h=3 slopes have a stud on top)
        run = brick_l - 1 if brick_h == 3 and brick_l > 1 else brick_l
        step_x, step_y = _physical_footprint(run, brick_w, rotation)

        x_min, x_max, y_min, y_max, z_min, z_max = _region_voxel_bounds(
            mesh, region, voxel_origin)

        n_x = max(1, round((x_max - x_min + 1) / step_x))
        n_y = max(1, round((y_max - y_min + 1) / step_y))

        if region_n_steps is not None:
            n_z = region_n_steps[region_idx]
        else:
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
            brick_l, brick_w, brick_h, foot_x, foot_y,
            step_x, step_y, rotation, world_shape
        )

        # Process top-down so higher steps clear voxels before lower steps check above
        placed.sort(key=lambda b: -b.z)
        accepted = []
        # Track rejection reasons for diagnostics
        rejected_bounds = rejected_no_voxels = rejected_noise = rejected_above = 0
        for brick in placed:
            sx, sy, sz = brick.slice
            if sx.stop > world_shape[0] or sy.stop > world_shape[1] or sz.stop > world_shape[2]:
                rejected_bounds += 1
                continue

            # Check if brick would overlap with already-placed slope bricks from other regions
            would_overlap = False
            for existing in slope_bricks:
                if existing.type != 1:
                    continue
                esx, esy, esz = existing.slice
                bsx, bsy, bsz = brick.slice
                if (esx.start < bsx.stop and bsx.start < esx.stop and
                        esy.start < bsy.stop and bsy.start < esy.stop and
                        esz.start < bsz.stop and bsz.start < esz.stop):
                    would_overlap = True
                    break
            if would_overlap:
                rejected_no_voxels += 1
                continue

            # Check if brick has no voxels - if so, verify it's supported from below
            if not remaining[brick.slice].any():
                sx, sy, sz = brick.slice
                if brick.z > 0:
                    voxels_below = voxels[sx, sy, brick.z - 1]
                    if not voxels_below.any():
                        # No voxels and no support below - floating brick
                        rejected_no_voxels += 1
                        continue
                else:
                    # At ground level with no voxels - reject
                    rejected_no_voxels += 1
                    continue

            # Brick has voxels - check noise in air triangle and voxels above
            slope_sl, stud_sl = _slope_region_slice(brick, region.slope_direction)
            slope_sx, slope_sy, slope_sz = slope_sl

            if stud_sl is not None:
                # h=3: check air triangle within bounding box (z+1 to z+h)
                air_start = brick.z + 1
                air_end = min(brick.z + brick_h, world_shape[2])
                noise_count = int(remaining[slope_sx, slope_sy, air_start:air_end].sum()) if air_start < air_end else 0
            else:
                # h=2: check one voxel above the bounding box
                z_above = brick.z + brick_h
                noise_count = int(remaining[slope_sx, slope_sy, z_above:z_above + 1].sum()) if z_above < world_shape[2] else 0

            # For h=3: the air triangle spans brick_h-1 voxels; all expected to be filled
            # For h=2: at most 1 stray voxel above is tolerable
            max_noise = (brick_h - 1) if stud_sl is not None else 1
            if noise_count > max_noise:
                rejected_noise += 1
                continue

            # Check the full column above the bbox in the sloping column
            # using the ORIGINAL voxel grid (not remaining, modified by higher steps)
            z_above = brick.z + brick_h
            above_count = int(voxels[slope_sx, slope_sy, z_above:].sum()) if z_above < world_shape[2] else 0
            if above_count > 1:
                rejected_above += 1
                continue

            # Clear the brick's voxels
            remaining[brick.slice] = False

            # Clear air triangle and single stray voxel above
            if 0 < noise_count <= max_noise:
                if stud_sl is not None:
                    remaining[slope_sx, slope_sy, air_start:air_end] = False
                else:
                    remaining[slope_sx, slope_sy, z_above:z_above + 1] = False
            if above_count == 1:
                remaining[slope_sx, slope_sy, z_above:] = False

            accepted.append(brick)

        final = accepted
        final = _merge_slope_bricks(final, region.slope_direction, matched_bricks, n_y if region.slope_direction in (0, 2) else n_x)
        slope_bricks.extend(final)

        # Print rejection diagnostics
        total_candidates = len(placed)
        if total_candidates > 0:
            print(f"    Region {region_idx}: {len(final)}/{total_candidates} placed "
                  f"(rejected: bounds={rejected_bounds}, no_voxels={rejected_no_voxels}, "
                  f"noise={rejected_noise}, above={rejected_above})")

    return slope_bricks, remaining
