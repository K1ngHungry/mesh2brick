import numpy as np
import open3d as o3d

from mesh2brick.data.brick_structure import BrickStructure
from mesh2brick.slopes import prepare_slopes, place_slope_bricks, SlopeConfig
from mesh2brick.voxel2brick import voxel2brick


def normalize_mesh(mesh, x_rotation: float = 90):
    # Translate the mesh to the origin
    mesh.translate(-mesh.get_center())

    # Scale the mesh to fit within a unit cube
    bbox = mesh.get_max_bound() - mesh.get_min_bound()
    scale_factor = 1 / np.max(bbox)
    mesh.scale(scale_factor, center=np.array([0, 0, 0]))

    x_rotation_radians = np.deg2rad(x_rotation)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((x_rotation_radians, 0, 0))
    rotated_mesh = mesh.rotate(rotation_matrix, center=mesh.get_center())

    return rotated_mesh


class Mesh2Brick:
    def __init__(
            self,
            world_dim: tuple[int, int, int] = (20, 20, 20), #change
            start_grid_shape: tuple[int, int, int] = (128, 128, 128),
            enable_slopes: bool = True,
            slope_config: SlopeConfig | None = None,
            **kwargs,
    ):
        self.world_dim = world_dim
        self.start_grid_shape = start_grid_shape
        self.enable_slopes = enable_slopes
        self.slope_config = slope_config if slope_config is not None else SlopeConfig()
        self.kwargs = kwargs

    def __call__(self, mesh, x_rotation: float = 90) -> BrickStructure:
        """
        :param mesh: A mesh object or a string, the filename of the input mesh.
            Any format Open3D can read is accepted (e.g. ``.obj``, ``.glb``).
        :return: The mesh converted to a brick structure. When ``enable_slopes``
            is set, sloped surfaces are detected, the mesh is deformed to align
            them to the voxel grid, and slope bricks are placed on those regions.
        """
        if isinstance(mesh, str):
            mesh = o3d.io.read_triangle_mesh(mesh)
        if self.enable_slopes:
            return self._mesh2brick_with_slopes(mesh, x_rotation=x_rotation)
        return voxel2brick(self.mesh2voxel(mesh, x_rotation=x_rotation), **self.kwargs)

    def _mesh2brick_with_slopes(self, mesh, x_rotation: float = 90) -> BrickStructure:
        """Slope-aware pipeline mirroring the Objaverse evaluation flow: detect
        sloped surfaces, deform the mesh to align them to the voxel grid, place
        slope bricks on those regions, then fill the remainder with standard
        bricks."""
        mesh = normalize_mesh(mesh, x_rotation=x_rotation)

        resolution = self.world_dim[0]
        slope_result = prepare_slopes(mesh, resolution=resolution, cfg=self.slope_config)

        # Voxelize the (deformed/scaled) mesh on a unit grid.
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(slope_result.mesh, 1.0)
        voxels = np.zeros(slope_result.world_dim, dtype=np.uint8)
        for voxel in np.asarray(voxel_grid.get_voxels()):
            idx = tuple(np.floor(voxel.grid_index).astype(int))
            if all(0 <= i < d for i, d in zip(idx, slope_result.world_dim)):
                voxels[idx] = 1

        # Place slope bricks on detected regions; fill the rest with standard bricks.
        if slope_result.assignments:
            voxel_origin = np.asarray(voxel_grid.origin)
            slope_bricks, remaining_voxels = place_slope_bricks(
                voxels, slope_result.mesh, slope_result.assignments, voxel_origin=voxel_origin,
            )
        else:
            slope_bricks, remaining_voxels = [], voxels

        bricks = voxel2brick(remaining_voxels, **self.kwargs)
        for brick in slope_bricks:
            bricks.add_brick(brick)
        return bricks

    def mesh2voxel(self, mesh, x_rotation: float = 90) -> np.ndarray:
        mesh = normalize_mesh(mesh, x_rotation=x_rotation)
        
        # Scale Z by 3 to compensate for plate height (1 unit) vs brick height (3 units)
        vertices = np.asarray(mesh.vertices)
        vertices[:, 2] *= 3.0
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        voxel_size = 0
        grid_shape = list(self.start_grid_shape)
        while (grid_shape[0] > self.world_dim[0] or 
               grid_shape[1] > self.world_dim[1] or 
               grid_shape[2] > self.world_dim[2]):
            voxel_size += 0.01
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
            voxel_indices = np.asarray(voxel_grid.get_voxels())
            min_bound = voxel_grid.get_min_bound()
            max_bound = voxel_grid.get_max_bound()
            grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

        voxel_array = np.zeros(self.world_dim, dtype=np.uint8)
        for voxel in voxel_indices:
            idx = np.floor(voxel.grid_index).astype(int)
            if all(0 <= i < d for i, d in zip(idx, self.world_dim)):
                voxel_array[tuple(idx)] = 1 

        return voxel_array
