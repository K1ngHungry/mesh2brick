import numpy as np
import open3d as o3d

from mesh2brick.data.brick_structure import BrickStructure
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
            world_dim: tuple[int, int, int] = (20, 20, 20),
            start_grid_shape: tuple[int, int, int] = (128, 128, 128),
            **kwargs,
    ):
        self.world_dim = world_dim
        self.start_grid_shape = start_grid_shape
        self.kwargs = kwargs

    def __call__(self, mesh, x_rotation: float = 90) -> BrickStructure:
        """
        :param mesh: A mesh object or a string, the filename of the input mesh.
        :return: The mesh converted to a brick structure.
        """
        if isinstance(mesh, str):
            mesh = o3d.io.read_triangle_mesh(mesh)
        bricks = voxel2brick(self.mesh2voxel(mesh, x_rotation=x_rotation), **self.kwargs)
        return bricks

    def mesh2voxel(self, mesh, x_rotation: float = 90) -> np.ndarray:
        mesh = normalize_mesh(mesh, x_rotation=x_rotation)

        voxel_size = 0
        grid_shape = list(self.start_grid_shape)
        while max(grid_shape) > max(self.world_dim):
            voxel_size += 0.01
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
            voxel_indices = np.asarray(voxel_grid.get_voxels())
            min_bound = voxel_grid.get_min_bound()
            max_bound = voxel_grid.get_max_bound()
            grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

        voxel_array = np.zeros(self.world_dim, dtype=np.uint8)
        for voxel in voxel_indices:
            idx = np.floor(voxel.grid_index).astype(int)
            voxel_array[tuple(idx)] = 1

        return voxel_array
