import math
import re
import warnings
from dataclasses import dataclass

import networkx as nx
import numpy as np

from mesh2brick.data.brick_library import (brick_library, dimensions_to_brick_id, brick_id_to_dimensions,
                                           brick_id_to_part_id, part_id_to_brick_id, brick_id_to_type)
from mesh2brick.stability_analysis.stability_analysis import StabilityConfig, stability_score


@dataclass(frozen=True, order=True, kw_only=True)
class Brick:
    """
    Represents a 1-unit-tall rectangular brick.
    """
    type: int = 0  # 0=Brick/Plate, 1=Slope
    l: int
    w: int
    h: int = 3
    rotation: int = 0  # 0-3
    x: int
    y: int
    z: int

    @property
    def brick_id(self) -> int:
        return dimensions_to_brick_id(self.type, self.l, self.w, self.h)

    @property
    def part_id(self) -> str:
        return brick_id_to_part_id(self.brick_id)

    @property
    def angle(self) -> float | None:
        """Slope angle in degrees, or None for non-slope bricks."""
        return None

    @property
    def area(self) -> int:
        return self.l * self.w

    @property
    def volume(self) -> int:
        return self.l * self.w * self.h

    @property
    def slice_2d(self) -> (slice, slice):
        # from_json already swaps l,w for odd rotations
        return slice(self.x, self.x + self.l), slice(self.y, self.y + self.w)

    @property
    def slice(self) -> (slice, slice, slice):
        return *self.slice_2d, slice(self.z, self.z + self.h)

    def overlaps(self, other: 'Brick') -> bool:
        """Check if this brick's 3D bounding box overlaps another's."""
        s1x, s1y, s1z = self.slice
        s2x, s2y, s2z = other.slice
        return (s1x.start < s2x.stop and s2x.start < s1x.stop and
                s1y.start < s2y.stop and s2y.start < s1y.stop and
                s1z.start < s2z.stop and s2z.start < s1z.stop)

    def __repr__(self):
        return self.to_txt()[:-1]

    def to_json(self) -> dict:
        return {
            'brick_id': self.brick_id,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'type': self.type,
            'rotation': self.rotation,
        }

    def to_txt(self) -> str:
        return f'T{self.type} {self.l}x{self.w}x{self.h} R{self.rotation} ({self.x},{self.y},{self.z})\n'

    def to_ldr(self, base_height: float = 0) -> str:
        x = (self.x + self.l * 0.5) * 20
        z = (self.y + self.w * 0.5) * 20
        y = (self.z + self.h + base_height) * -8
        brick_matrices = [
            '0 0 1 0 1 0 -1 0 0',      # R0 (270 deg)
            '-1 0 0 0 1 0 0 0 -1',     # R1 (180 deg)
            '0 0 -1 0 1 0 1 0 0',      # R2 (90 deg)
            '1 0 0 0 1 0 0 0 1'        # R3 (0 deg / identity)
        ]
        matrix = brick_matrices[self.rotation % 4]
        line = f'1 115 {x} {y} {z} {matrix} {self.part_id}\n'
        step_line = '0 STEP\n'
        return line + step_line

    @classmethod
    def from_json(cls, brick_json: dict):
        brick_type = brick_id_to_type(brick_json['brick_id'])
        l, w, h = brick_id_to_dimensions(brick_json['brick_id'])
        rotation = brick_json['rotation']
        if brick_type == 1:
            slope_dir = _ROTATION_TO_SLOPE_DIR[rotation]
            return SlopeBrick(l=l, w=w, h=h, slope_direction=slope_dir, x=brick_json['x'], y=brick_json['y'], z=brick_json['z'])
        if rotation % 2 == 1:
            l, w = w, l
        x, y, z = brick_json['x'], brick_json['y'], brick_json['z']
        return cls(type=brick_type, l=l, w=w, h=h, rotation=rotation, x=x, y=y, z=z)

    @classmethod
    def from_txt(cls, brick_txt: str):
        brick_txt = brick_txt.strip()
        match = re.fullmatch(r'T(\d+) (\d+)x(\d+)x(\d+) R(\d+) \((\d+),(\d+),(\d+)\)', brick_txt)
        if match is None:
            raise ValueError(f'Text Format brick is ill-formatted: {brick_txt}')
        brick_type = int(match.group(1))
        l, w, h = map(int, match.group(2, 3, 4))
        rotation = int(match.group(5))
        x, y, z = map(int, match.group(6, 7, 8))
        if brick_type == 1:
            return SlopeBrick(l=l, w=w, h=h, slope_direction=_ROTATION_TO_SLOPE_DIR[rotation], x=x, y=y, z=z)
        return cls(type=brick_type, l=l, w=w, h=h, rotation=rotation, x=x, y=y, z=z)

    @classmethod
    def from_ldr(cls, brick_ldr: str):
        ldr_components = brick_ldr.strip().split()
        match ldr_components:
            case ['1', _, x0, y0, z0, *matrix, part_id]:
                x0, y0, z0 = map(float, (x0, y0, z0))
                matrix_str = ' '.join(matrix)

                l, w, h = brick_id_to_dimensions(part_id_to_brick_id(part_id))
                brick_type = brick_id_to_type(part_id_to_brick_id(part_id))

                brick_matrix_to_rotation = {
                    '0 0 1 0 1 0 -1 0 0': 0,
                    '-1 0 0 0 1 0 0 0 -1': 1,
                    '0 0 -1 0 1 0 1 0 0': 2,
                    '1 0 0 0 1 0 0 0 1': 3,
                }
                slope_matrix_to_rotation = {
                    '1 0 0 0 1 0 0 0 1': 0,
                    '0 0 -1 0 1 0 1 0 0': 1,
                    '-1 0 0 0 1 0 0 0 -1': 2,
                    '0 0 1 0 1 0 -1 0 0': 3,
                }
                matrix_map = slope_matrix_to_rotation if brick_type == 1 else brick_matrix_to_rotation
                if matrix_str not in matrix_map:
                    raise ValueError(f'Invalid transformation matrix: {matrix_str}')
                rotation = matrix_map[matrix_str]
                if brick_type != 1 and rotation % 2 == 1:
                    l, w = w, l

                if brick_type == 1:
                    if rotation % 2 == 0:
                        x = int(x0 / 20 - w * 0.5)
                        y = int(z0 / 20 - l * 0.5)
                    else:
                        x = int(x0 / 20 - l * 0.5)
                        y = int(z0 / 20 - w * 0.5)
                else:
                    x = int(x0 / 20 - l * 0.5)
                    y = int(z0 / 20 - w * 0.5)
                z = int(-y0 / 8 - h)

                if brick_type == 1:
                    return SlopeBrick(l=l, w=w, h=h, slope_direction=_ROTATION_TO_SLOPE_DIR[rotation], x=x, y=y, z=z)
                return cls(type=brick_type, l=l, w=w, h=h, rotation=rotation, x=x, y=y, z=z)
            case _:
                raise ValueError(f"LDR format is ill-formatted: {brick_ldr}")


_SLOPE_DIR_TO_ROTATION = {0: 1, 1: 2, 2: 3, 3: 0}
_ROTATION_TO_SLOPE_DIR = {v: k for k, v in _SLOPE_DIR_TO_ROTATION.items()}


@dataclass(frozen=True, order=True, kw_only=True)
class SlopeBrick(Brick):
    """A slope brick that knows its slope direction.

    rotation is auto-derived from slope_direction and should not be passed.
    """
    slope_direction: int  # 0=+X, 1=+Y, 2=-X, 3=-Y

    def __post_init__(self):
        object.__setattr__(self, 'type', 1)
        object.__setattr__(
            self, 'rotation', _SLOPE_DIR_TO_ROTATION[self.slope_direction])

    @property
    def angle(self) -> float:
        """Slope angle in degrees (atan(h/l))."""
        return math.degrees(math.atan(self.h / self.l))

    @property
    def slice_2d(self) -> (slice, slice):
        # Slope run (l) maps to LDR Z (our Y) for even rotation,
        # and to LDR X (our X) for odd rotation.
        if self.rotation % 2 == 0:
            return slice(self.x, self.x + self.w), slice(self.y, self.y + self.l)
        return slice(self.x, self.x + self.l), slice(self.y, self.y + self.w)

    @staticmethod
    def rotated_dim(l: int, w: int, rotation: int) -> tuple[int, int]:
        """Return (dim_x, dim_y) for a slope brick's l/w given rotation."""
        if rotation % 2 == 0:
            return w, l
        return l, w

    @property
    def rotated_dims(self) -> tuple[int, int]:
        """Return (dim_x, dim_y) for this brick."""
        return SlopeBrick.rotated_dim(self.l, self.w, self.rotation)

    def slope_slice(
        self,
    ) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice] | None]:
        """Split this slope brick's slice into its sloping surface and flat stud column.

        For h=3 bricks with l>1, the first l-1 columns (along the slope axis) are
        the slope and the last column is a full-height stud. The stud portion can
        connect to bricks above while the slope portion cannot. For h=2 or l=1
        bricks, the entire footprint is slope and stud_slice is None.
        """
        sx, sy, sz = self.slice
        run = self.l - 1 if self.h == 3 and self.l > 1 else self.l
        if run == self.l:
            return (sx, sy, sz), None

        if self.slope_direction in (0, 2):
            if self.slope_direction == 0:  # +X: slope at low x, stud at high x
                slope_x = slice(sx.start, sx.start + run)
                stud_x = slice(sx.start + run, sx.stop)
            else:  # -X: slope at high x, stud at low x
                slope_x = slice(sx.stop - run, sx.stop)
                stud_x = slice(sx.start, sx.stop - run)
            return (slope_x, sy, sz), (stud_x, sy, sz)
        else:
            if self.slope_direction == 1:  # +Y: slope at low y, stud at high y
                slope_y = slice(sy.start, sy.start + run)
                stud_y = slice(sy.start + run, sy.stop)
            else:  # -Y: slope at high y, stud at low y
                slope_y = slice(sy.stop - run, sy.stop)
                stud_y = slice(sy.start, sy.stop - run)
            return (sx, slope_y, sz), (sx, stud_y, sz)

    def to_ldr(self, base_height: float = 0) -> str:
        # Slope run (l) maps to LDR Z (our Y) for even rotation,
        # and to LDR X (our X) for odd rotation.
        if self.h == 2:
            # Cheese slopes: centered formula, origin at bottom
            if self.rotation % 2 == 0:
                x = (self.x + self.w * 0.5) * 20
                z = (self.y + self.l * 0.5) * 20
            else:
                x = (self.x + self.l * 0.5) * 20
                z = (self.y + self.w * 0.5) * 20
            y = (self.z + base_height) * -8
        else:
            # h=3 slopes: offset formula, origin at top
            if self.rotation == 0:
                x = (self.x + self.w * 0.5) * 20
                z = (self.y + self.l) * 20 - 10
            elif self.rotation == 2:
                x = (self.x + self.w * 0.5) * 20
                z = self.y * 20 + 10
            elif self.rotation == 1:
                x = self.x * 20 + 10
                z = (self.y + self.w * 0.5) * 20
            elif self.rotation == 3:
                x = (self.x + self.l) * 20 - 10
                z = (self.y + self.w * 0.5) * 20
            y = (self.z + self.h + base_height) * -8
        slope_matrices = [
            '1 0 0 0 1 0 0 0 1',       # R0 (identity)
            '0 0 -1 0 1 0 1 0 0',      # R1 (90 deg)
            '-1 0 0 0 1 0 0 0 -1',     # R2 (180 deg)
            '0 0 1 0 1 0 -1 0 0',      # R3 (270 deg)
        ]
        matrix = slope_matrices[self.rotation % 4]
        line = f'1 115 {x} {y} {z} {matrix} {self.part_id}\n'
        step_line = '0 STEP\n'
        return line + step_line


class BrickStructure:
    """
    Represents a brick structure in the form of a list of bricks.
    """

    def __init__(self, bricks: list[Brick], world_dim: int | tuple[int, int, int] = 20):
        if isinstance(world_dim, int):
            self.world_dim = (world_dim, world_dim, world_dim)
        else:
            self.world_dim = world_dim

        # Check if structure starts at ground level
        z0 = min((brick.z for brick in bricks), default=0)
        if z0 != 0:
            warnings.warn('Brick structure does not start at ground level z=0.')

        # Build structure from bricks
        self.bricks = []
        self.voxel_occupancy = np.zeros(self.world_dim, dtype=int)
        for brick in bricks:
            self.add_brick(brick)

    def __len__(self):
        return len(self.bricks)

    def __repr__(self):
        return self.to_txt()

    def __eq__(self, other) -> bool:
        if not isinstance(other, BrickStructure):
            return NotImplemented
        return self.bricks == other.bricks

    def to_json(self) -> dict:
        return {str(i + 1): brick.to_json() for i, brick in enumerate(self.bricks)}

    def to_txt(self) -> str:
        return ''.join([brick.to_txt() for brick in self.bricks])

    def to_ldr(self) -> str:
        return ''.join([brick.to_ldr() for brick in self.bricks])

    def add_brick(self, brick: Brick) -> None:
        self.bricks.append(brick)
        self.voxel_occupancy[brick.slice] += 1

    def undo_add_brick(self) -> None:
        brick = self.bricks[-1]
        self.voxel_occupancy[brick.slice] -= 1
        self.bricks.pop()

    def has_out_of_bounds_bricks(self) -> bool:
        return any(not self.brick_in_bounds(brick) for brick in self.bricks)

    def brick_in_bounds(self, brick: Brick) -> bool:
        return (all(slice_.start >= 0 and slice_.stop <= self.world_dim[i] for i, slice_ in enumerate(brick.slice_2d))
                and 0 <= brick.z and brick.z + brick.h <= self.world_dim[2])

    def has_collisions(self) -> bool:
        return np.any(self.voxel_occupancy > 1)

    def brick_collides(self, brick: Brick) -> bool:
        return np.any(self.voxel_occupancy[brick.slice])

    def has_floating_bricks(self) -> bool:
        return any(self.brick_floats(brick) for brick in self.bricks)

    def brick_floats(self, brick: Brick) -> bool:
        if brick.z == 0:
            return False  # Supported by ground
        if np.any(self.voxel_occupancy[brick.slice_2d[0], brick.slice_2d[1], brick.z - 1]):
            return False  # Supported from below
        if brick.z + brick.h < self.world_dim[2] and np.any(
                self.voxel_occupancy[brick.slice_2d[0], brick.slice_2d[1], brick.z + brick.h]):
            return False  # Supported from above
        return True

    def is_stable(self) -> bool:
        if self.has_floating_bricks() or self.has_collisions():
            return False
        return self.stability_scores().max() < 1

    def stability_scores(self) -> np.ndarray:
        scores, _ = self.stability_scores_with_status()
        return scores

    def stability_scores_with_status(self) -> tuple[np.ndarray, bool]:
        if self.has_collisions():
            raise ValueError('Cannot compute stability scores - structure has colliding bricks.')
        if self.has_out_of_bounds_bricks():
            raise ValueError('Cannot compute stability scores - structure has out of bounds bricks.')
        scores, _, _, _, _, solver_optimal = stability_score(self.to_json(), brick_library,
                                                                 StabilityConfig(world_dimension=self.world_dim))
        return scores, solver_optimal

    @classmethod
    def from_json(cls, bricks_json: dict, world_dim: int | tuple[int, int, int] = 20):
        bricks = [Brick.from_json(v) for k, v in bricks_json.items() if k.isdigit()]
        return cls(bricks, world_dim=world_dim)

    @classmethod
    def from_txt(cls, bricks_txt: str, world_dim: int | tuple[int, int, int] = 20):
        bricks_txt = bricks_txt.split('\n')
        bricks_txt = [b for b in bricks_txt if b.strip()]  # Remove blank lines
        bricks = [Brick.from_txt(brick) for brick in bricks_txt]
        return cls(bricks, world_dim=world_dim)

    @classmethod
    def from_ldr(cls, bricks_ldr: str, world_dim: int | tuple[int, int, int] = 20):
        bricks_ldr = bricks_ldr.split('0 STEP')  # Split on step lines
        bricks_ldr = [b for b in bricks_ldr if b.strip()]  # Remove blank or whitespace-only lines
        bricks = [Brick.from_ldr(brick) for brick in bricks_ldr]
        return cls(bricks, world_dim=world_dim)


class ConnectivityBrickStructure:
    """
    Brick structure that keeps graph connectivity information
    """

    def __init__(self, shape: tuple[int, int, int]):
        self.voxel_bricks = np.zeros(shape, dtype=int)  # Which brick occupies each voxel; 0 = no brick
        self.bricks = {}  # Dictionary node_id -> brick
        self.node_id_counter = 0

        self.connection_graph = nx.Graph()
        self.neighbor_graph = nx.Graph()

        self._connected_components = None
        self._component_labels = None
        self._node2component = None

    @property
    def max_x(self) -> int:
        return self.voxel_bricks.shape[0]

    @property
    def max_y(self) -> int:
        return self.voxel_bricks.shape[1]

    @property
    def max_z(self) -> int:
        return self.voxel_bricks.shape[2]

    @property
    def voxels(self) -> np.ndarray:
        return self.voxel_bricks != 0

    def _reset_cache(self) -> None:
        self._connected_components = None
        self._component_labels = None
        self._node2component = None

    def n_components(self) -> int:
        return len(self.connected_components())

    def connected_components(self):
        if self._connected_components is None:
            self._connected_components = list(nx.connected_components(self.connection_graph))
        return self._connected_components

    def component_labels(self) -> np.ndarray:
        if self._component_labels is None:
            self._component_labels = np.zeros_like(self.voxel_bricks)
            for i, comp in enumerate(self.connected_components()):
                for node in comp:
                    brick = self.bricks[node]
                    self._component_labels[brick.slice] = i + 1
        return self._component_labels

    def node2component(self) -> dict[int, int]:
        if self._node2component is None:
            self._node2component = {node: component_idx + 1
                                    for component_idx, component in enumerate(self.connected_components())
                                    for node in component}
        return self._node2component

    def stability_score(self) -> tuple[np.ndarray, bool]:
        bricks = BrickStructure(list(self.bricks.values()), self.voxel_bricks.shape)
        return bricks.stability_scores_with_status()

    def node_exists(self, node_id: int):
        return node_id in self.bricks

    def add_brick(self, brick: Brick) -> int:
        self._reset_cache()

        if self.voxel_bricks[brick.slice].any():  # Brick overlaps other bricks on layer
            raise ValueError(f'Cannot place brick {brick} due to collisions')

        self.node_id_counter += 1
        node = self.node_id_counter
        self.bricks[node] = brick
        self.voxel_bricks[brick.slice] = node

        # Update graph edges
        self.connection_graph.add_node(node)
        self.neighbor_graph.add_node(node)
        vert_neighbors = ({(node, self.voxel_bricks[x, y, brick.z - 1])
                           for x in range(brick.x, brick.x + brick.l) for y in range(brick.y, brick.y + brick.w)
                           if brick.z > 0} |
                          {(node, self.voxel_bricks[x, y, brick.z + brick.h])
                           for x in range(brick.x, brick.x + brick.l) for y in range(brick.y, brick.y + brick.w)
                           if brick.z + brick.h < self.max_z})
        vert_neighbors = list(filter(lambda e: e[1] != 0, vert_neighbors))  # Remove connections with empty bricks
        horz_neighbors = set()
        for z in range(brick.z, brick.z + brick.h):
            horz_neighbors |= ({(node, self.voxel_bricks[brick.x - 1, y, z])
                                for y in range(brick.y, brick.y + brick.w) if brick.x > 0} |
                               {(node, self.voxel_bricks[brick.x + brick.l, y, z])
                                for y in range(brick.y, brick.y + brick.w) if brick.x + brick.l < self.max_x} |
                               {(node, self.voxel_bricks[x, brick.y - 1, z])
                                for x in range(brick.x, brick.x + brick.l) if brick.y > 0} |
                               {(node, self.voxel_bricks[x, brick.y + brick.w, z])
                                for x in range(brick.x, brick.x + brick.l) if brick.y + brick.w < self.max_y})
        horz_neighbors = list(filter(lambda e: e[1] != 0, horz_neighbors))  # Remove connections with empty bricks
        self.connection_graph.add_edges_from(vert_neighbors)
        self.neighbor_graph.add_edges_from(vert_neighbors + horz_neighbors)

        return node

    def add_bricks(self, bricks: list[Brick]) -> list[int]:
        return [self.add_brick(brick) for brick in bricks]

    def remove_brick(self, node_id: int) -> None:
        self._reset_cache()

        brick = self.bricks[node_id]
        self.bricks.pop(node_id)
        self.voxel_bricks[brick.slice] = 0
        self.connection_graph.remove_node(node_id)
        self.neighbor_graph.remove_node(node_id)

    def remove_voxel_subset(self, voxel_subset: np.ndarray) -> list[Brick]:
        """
        Erases all bricks inside the specified subset of voxels.
        Assumes that all bricks in voxel_subset are completely contained within voxel_subset.
        """
        removed_bricks = []
        nodes = set(np.unique(self.voxel_bricks[voxel_subset])) - {0}
        for node in nodes:
            brick = self.bricks[node]
            assert voxel_subset[brick.slice].all()
            removed_bricks.append(brick)
            self.remove_brick(node)
        return removed_bricks
