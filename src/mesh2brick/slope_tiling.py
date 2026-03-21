"""Place slope bricks in a voxel grid based on detected slope regions."""

import numpy as np
import open3d as o3d
from queue import PriorityQueue

from mesh2brick.data.brick_library import brick_library
from mesh2brick.data.brick_structure import Brick
from mesh2brick.slope_detection import SlopeRegion
from mesh2brick.voxel2brick import get_merged_brick


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
    maxs = np.ceil(voxel_coords.max(axis=0)).astype(int)
    return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]


def _slope_direction_to_rotation(slope_direction: int) -> int:
    """Map slope_direction (0=+X, 1=+Y, 2=-X, 3=-Y) to brick rotation."""
    mapping = {0: 1, 1: 2, 2: 3, 3: 0}
    return mapping[slope_direction]


def _physical_footprint(brick_l: int, brick_w: int, rotation: int) -> tuple[int, int]:
    if rotation % 2 == 0:
        return brick_w, brick_l
    return brick_l, brick_w


def _greedy_merge_slopes(bricks_1x1: list[Brick]) -> list[Brick]:
    """Merges 1x1 slope voxels greedily within each Z layer."""
    merged_results = []
    by_z = {}
    for b in bricks_1x1:
        by_z.setdefault(b.z, []).append(b)
        
    for z, layer_bricks in by_z.items():
        nodes = {i: b for i, b in enumerate(layer_bricks)}
        pq = PriorityQueue()
        
        def add_pairs(n1):
            b1 = nodes[n1]
            for n2, b2 in nodes.items():
                if n1 >= n2: continue
                merged = get_merged_brick(b1, b2)
                if merged:
                    pq.put((-merged.volume, n1, n2, merged))
                    
        for n in list(nodes.keys()):
            add_pairs(n)
            
        while not pq.empty():
            _, n1, n2, merged_brick = pq.get()
            if n1 not in nodes or n2 not in nodes:
                continue
            del nodes[n1]
            del nodes[n2]
            n3 = max(nodes.keys(), default=-1) + 1
            nodes[n3] = merged_brick
            for n2, b2 in nodes.items():
                if n3 == n2: continue
                merged = get_merged_brick(merged_brick, b2)
                if merged:
                    pq.put((-merged.volume, n3, n2, merged))
                    
        merged_results.extend(nodes.values())
    return merged_results


def _tile_rigid_block(
    slope_direction: int,
    x_min: int, y_min: int, z_min: int,
    n_x: int, n_y: int, n_z: int,
    brick_l: int, brick_w: int, brick_h: int,
    foot_x: int, foot_y: int,
    rotation: int,
    world_shape: tuple[int, ...]
) -> list[Brick]:
    """Strictly construct the n_w x n_h panel of stacked bricks."""
    placed = []
    for step_z in range(n_z):
        z = z_min + step_z * brick_h
        if z + brick_h > world_shape[2]:
            continue
            
        # The staircasing logic exactly maps the Z height strictly to the XY offset:
        if slope_direction == 0:  # +X slope rising
            start_x = x_min + (n_z - 1 - step_z) * foot_x
            n_step_x = 1
            n_step_y = n_y
        elif slope_direction == 2:  # -X slope rising
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
                x = (start_x if slope_direction in (0,2) else x_min + ix * foot_x)
                y = (start_y if slope_direction in (1,3) else y_min + iy * foot_y)
                
                if x >= 0 and y >= 0 and x + foot_x <= world_shape[0] and y + foot_y <= world_shape[1]:
                    placed.append(Brick(type=1, l=brick_l, w=brick_w, h=brick_h, rotation=rotation, x=int(x), y=int(y), z=int(z)))
    return placed


def place_slope_bricks(
    voxels: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    assignments: list[tuple[SlopeRegion, list[dict]]],
    irregular_assignments: list[tuple[SlopeRegion, list[dict]]] | None = None,
    voxel_origin: np.ndarray | None = None,
) -> tuple[list[Brick], np.ndarray]:
    
    if irregular_assignments is None:
        irregular_assignments = []

    remaining = voxels.astype(bool).copy()
    slope_bricks: list[Brick] = []
    world_shape = voxels.shape

    if voxel_origin is None:
        voxel_origin = np.asarray(mesh.get_min_bound())

    # 1. Tile strictly rectangular regions
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

        # n_w and n_h based exactly on bounding box bounds generated by _resize_region round() algorithm
        n_x = max(1, round((x_max - x_min) / foot_x))
        n_y = max(1, round((y_max - y_min) / foot_y))
        n_z = max(1, round((z_max - z_min) / brick_h))

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
                z_above_start = brick.z + brick_h
                z_above_end = min(world_shape[2], z_above_start + brick_h)
                if z_above_start < z_above_end:
                    remaining[sx, sy, z_above_start:z_above_end] = False
        slope_bricks.extend(placed)

    # 2. Tile irregular regions with 1x1 slope voxels & greedy re-merge
    for region, matched_bricks in irregular_assignments:
        if not matched_bricks:
            continue
            
        triangles = np.asarray(mesh.triangles)[region.face_indices]
        vertex_indices = np.unique(triangles)
        
        sub_mesh = o3d.geometry.TriangleMesh()
        vert_map = {old: new for new, old in enumerate(vertex_indices)}
        new_triangles = np.vectorize(vert_map.get)(triangles)
        sub_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[vertex_indices])
        sub_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        
        sub_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(sub_mesh, 1.0)
        sub_voxels = np.asarray(sub_grid.get_voxels())
        
        rotation = _slope_direction_to_rotation(region.slope_direction)
        placed_1x1s = []
        
        for voxel in sub_voxels:
            sub_coord = voxel.grid_index * 1.0 + np.asarray(sub_grid.origin)
            idx = np.floor(sub_coord - voxel_origin).astype(int)
            x, y, z = idx
            
            if 0 <= x < world_shape[0] and 0 <= y < world_shape[1] and 0 <= z < world_shape[2]:
                if remaining[x, y, z]:
                    placed_1x1s.append(Brick(type=1, l=1, w=1, h=1, rotation=rotation, x=x, y=y, z=z))
                    remaining[x, y, z] = False

        merged = _greedy_merge_slopes(placed_1x1s)
        for brick in merged:
            slope_bricks.append(brick)
            sx, sy, sz = brick.slice
            z_above_start = brick.z + brick.h
            z_above_end = min(world_shape[2], z_above_start + brick.h)
            if z_above_start < z_above_end:
                remaining[sx, sy, z_above_start:z_above_end] = False

    return slope_bricks, remaining
