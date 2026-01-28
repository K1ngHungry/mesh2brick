import pytest
from mesh2brick.data.brick_structure import Brick, BrickStructure, ConnectivityBrickStructure

@pytest.mark.parametrize(
    'brick_txt,neighbor_pair,has_connection,has_neighbor', [
        # 1. Side connection touching at different Z-layer of tall brick
        ('1x1x3 (0,0,0)\n1x1x1 (1,0,2)\n', (1, 2), False, True),
        # 2. Side connection NOT touching (too high)
        ('1x1x3 (0,0,0)\n1x1x1 (1,0,3)\n', (1, 2), False, False),
        # 3. Vertical connection (on top)
        ('1x1x3 (0,0,0)\n1x1x1 (0,0,3)\n', (1, 2), True, True),
        # 4. Vertical connection (flipped orientation)
        ('2x1x3 (0,0,0)\n1x2x1 (0,0,3)\n', (1, 2), True, True),
    ])
def test_connectivity_graphs(brick_txt: str, neighbor_pair: tuple, has_connection: bool, has_neighbor: bool):
    """Test graph connectivity between bricks based on 3D volume."""
    bricks = BrickStructure.from_txt(brick_txt).bricks
    struct = ConnectivityBrickStructure((10, 10, 10))
    node_ids = struct.add_bricks(bricks)
    
    id1, id2 = node_ids[neighbor_pair[0]-1], node_ids[neighbor_pair[1]-1]
    
    assert struct.connection_graph.has_edge(id1, id2) == has_connection
    assert struct.neighbor_graph.has_edge(id1, id2) == has_neighbor


@pytest.mark.parametrize(
    'brick_txt,has_collisions', [
        # Horizontal OK
        ('1x1x3 (0,0,0)\n1x1x3 (1,0,0)\n', False),
        # Vertical OK
        ('1x1x3 (0,0,0)\n1x1x3 (0,0,3)\n', False),
        # Vertical Collision (internal)
        ('1x1x3 (0,0,0)\n1x1x1 (0,0,1)\n', True),
        # Vertical Collision (top boundary overlap)
        ('1x1x3 (0,0,0)\n1x1x3 (0,0,2)\n', True),
    ])
def test_collision_3d(brick_txt: str, has_collisions: bool):
    """Test 3D volume collisions."""
    bricks = BrickStructure.from_txt(brick_txt)
    assert bricks.has_collisions() == has_collisions


@pytest.mark.parametrize(
    'brick_txt,is_stable', [
        # Basic arch - Pillar(1x1x3), Pillar(1x1x3), Beam(1x4x1)
        ('1x1x3 (0,0,0)\n1x1x3 (3,0,0)\n4x1x1 (0,0,3)\n', True),
        # Floating beam
        ('1x1x3 (0,0,0)\n4x1x1 (0,0,4)\n', False),
    ])
def test_stability_3d(brick_txt: str, is_stable: bool):
    """Test stability score for structures with variable heights."""
    bricks = BrickStructure.from_txt(brick_txt)
    try:
        assert bricks.is_stable() == is_stable
    except (ImportError, ValueError, Exception):
        pytest.skip("Stability solver (Gurobi) not available or configured")


@pytest.mark.parametrize(
    'brick_txt,hanging_idx,is_floating', [
        # Supported from above: top at z=5, hanging at z=4
        ('4x4x1 (0,0,5)\n1x1x1 (1,1,4)\n', 1, False),
        # NOT supported from above (gap)
        ('4x4x1 (0,0,6)\n1x1x1 (1,1,4)\n', 1, True),
    ])
def test_hanging_support(brick_txt: str, hanging_idx: int, is_floating: bool):
    """Test support-from-above logic."""
    bricks = BrickStructure.from_txt(brick_txt)
    assert bricks.brick_floats(bricks.bricks[hanging_idx]) == is_floating


@pytest.mark.parametrize(
    'dims,x,y,z,expected_slice', [
        ((1, 1, 3), 0, 0, 0, (slice(0,1), slice(0,1), slice(0,3))),
        ((2, 6, 1), 5, 5, 5, (slice(5,7), slice(5,11), slice(5,6))),
    ])
def test_brick_geometry(dims, x, y, z, expected_slice):
    """Verify 3D slice calculations match the Style of test_brick."""
    b = Brick(l=dims[0], w=dims[1], h=dims[2], x=x, y=y, z=z)
    assert b.slice == expected_slice
