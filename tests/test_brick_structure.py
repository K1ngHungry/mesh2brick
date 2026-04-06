import pytest
from mesh2brick.data.brick_structure import Brick, SlopeBrick, BrickStructure


def test_brick():
    brick_txt = 'T0 6x2x3 R1 (0,1,2)\n'
    brick_json = {'brick_id': 1, 'x': 0, 'y': 1, 'z': 2, 'type': 0, 'rotation': 1}

    for brick in [Brick.from_json(brick_json), Brick.from_txt(brick_txt)]:
        assert brick.brick_id == 1  # ID 1 is 2x6
        assert brick.h == 3
        assert brick.rotation == 1
        assert brick.area == 12     # 6*2 = 12
        assert brick.slice_2d == (slice(0, 6), slice(1, 3))
        assert brick.slice == (slice(0, 6), slice(1, 3), slice(2, 5))
        assert brick.to_json() == brick_json
        assert brick.to_txt() == brick_txt


def test_brick_structure():
    # 2x6 bricks at (0,0,0) and (2,0,0)
    bricks_txt = 'T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (2,0,0)\n'
    bricks_json = {
        '1': {'brick_id': 1, 'x': 0, 'y': 0, 'z': 0, 'type': 0, 'rotation': 0},
        '2': {'brick_id': 1, 'x': 2, 'y': 0, 'z': 0, 'type': 0, 'rotation': 0},
    }
    # R0 = 270° matrix for regular bricks, 2x6 brick: x_center=1*20=20, z_center=3*20=60, y=-(3)*8=-24
    bricks_ldr = '1 115 20.0 -24 60.0 0 0 1 0 1 0 -1 0 0 2456.DAT\n0 STEP\n' \
                 '1 115 60.0 -24 60.0 0 0 1 0 1 0 -1 0 0 2456.DAT\n0 STEP\n'

    for bricks in [BrickStructure.from_json(bricks_json), BrickStructure.from_txt(bricks_txt),
                   BrickStructure.from_ldr(bricks_ldr)]:
        assert len(bricks) == 2
        assert bricks.to_json() == bricks_json
        assert bricks.to_txt() == bricks_txt
        assert bricks.to_ldr() == bricks_ldr

    assert BrickStructure.from_txt(bricks_txt) == BrickStructure.from_json(bricks_json)
    assert BrickStructure.from_txt(bricks_txt) == BrickStructure.from_ldr(bricks_ldr)
    assert BrickStructure.from_txt(bricks_txt) != BrickStructure([])


def test_slope_brick():
    # 2x2x3 slope (brick_id 201, partID 3039.DAT)
    brick_txt = 'T1 2x2x3 R0 (0,0,0)\n'
    brick_json = {'brick_id': 201, 'x': 0, 'y': 0, 'z': 0, 'type': 1, 'rotation': 0}

    for brick in [Brick.from_json(brick_json), Brick.from_txt(brick_txt)]:
        assert isinstance(brick, SlopeBrick)
        assert brick.brick_id == 201
        assert brick.type == 1
        assert brick.l == 2
        assert brick.w == 2
        assert brick.h == 3
        assert brick.rotation == 0
        assert brick.slope_direction == 3  # rotation 0 -> slope_direction 3 (-Y)
        assert brick.to_json() == brick_json
        assert brick.to_txt() == brick_txt


def test_slope_brick_rotated():
    # 2x4x3 slope rotated — must keep brick_id 202, NOT become 205 (4x2 slope)
    brick_txt = 'T1 2x4x3 R1 (0,0,0)\n'
    brick_json = {'brick_id': 202, 'x': 0, 'y': 0, 'z': 0, 'type': 1, 'rotation': 1}

    for brick in [Brick.from_json(brick_json), Brick.from_txt(brick_txt)]:
        assert isinstance(brick, SlopeBrick)
        assert brick.brick_id == 202
        assert brick.type == 1
        assert brick.l == 2
        assert brick.w == 4
        assert brick.rotation == 1
        assert brick.slope_direction == 0  # rotation 1 -> slope_direction 0 (+X)
        assert brick.to_json() == brick_json
        assert brick.to_txt() == brick_txt


def test_slope_roundtrip_ldr():
    # 2x2x3 slope: x_center=1*20=20, z_center=1*20=20, y=-(3)*8=-24
    brick_ldr = '1 115 20.0 -24 20.0 1 0 0 0 1 0 0 0 1 3039.DAT'
    brick = Brick.from_ldr(brick_ldr)
    assert isinstance(brick, SlopeBrick)
    assert brick.type == 1
    assert brick.brick_id == 201
    assert brick.rotation == 0
    assert brick.x == 0
    assert brick.y == 0
    assert brick.z == 0
    assert brick.to_ldr() == brick_ldr + '\n0 STEP\n'


def test_slope_structure():
    # Slope on top of a regular brick
    bricks_txt = 'T0 2x2x3 R0 (0,0,0)\nT1 2x2x3 R0 (0,0,3)\n'
    bricks = BrickStructure.from_txt(bricks_txt)
    assert len(bricks) == 2
    assert bricks.bricks[0].type == 0
    assert bricks.bricks[1].type == 1
    assert not bricks.has_collisions()
    assert not bricks.has_floating_bricks()


@pytest.mark.parametrize(
    'brick_txt,has_collisions', [
        ('T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (2,0,0)\n', False),
        ('T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (1,0,0)\n', True),
    ])
def test_collision_check(brick_txt: str, has_collisions: bool):
    bricks = BrickStructure.from_txt(brick_txt)
    assert bricks.has_collisions() == has_collisions


@pytest.mark.parametrize(
    'brick_txt,has_floating_bricks', [
        ('T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (2,0,0)\n', False),
        ('T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (2,0,1)\n', True),
    ])
def test_floating_check(brick_txt: str, has_floating_bricks: bool):
    bricks = BrickStructure.from_txt(brick_txt)
    assert bricks.has_floating_bricks() == has_floating_bricks


@pytest.mark.parametrize(
    'brick_txt,is_stable', [
        pytest.param('T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (2,0,0)\n', True,
                     marks=pytest.mark.xfail(reason="stability_analysis still uses 'ori' key")),
        ('T0 2x6x3 R0 (0,0,0)\nT0 2x6x3 R0 (2,0,1)\n', False),
    ])
def test_stability_check(brick_txt: str, is_stable: bool):
    bricks = BrickStructure.from_txt(brick_txt)
    assert bricks.is_stable() == is_stable


@pytest.mark.parametrize(
    'brick_txt,is_in_bounds', [
        ('T0 2x6x3 R0 (0,0,0)\n', True),
        ('T0 2x6x3 R0 (18,0,0)\n', True),
        ('T0 2x6x3 R0 (19,0,0)\n', False),
    ])
def test_in_bounds(brick_txt: str, is_in_bounds: bool):
    bricks = BrickStructure([], world_dim=20)
    brick = Brick.from_txt(brick_txt)
    assert bricks.brick_in_bounds(brick) == is_in_bounds
