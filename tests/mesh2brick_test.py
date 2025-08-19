from pathlib import Path

import pytest

from mesh2brick.mesh2brick import Mesh2Brick


@pytest.mark.parametrize(
    'input_file, solution_file', [
        ('car.obj', 'car.txt'),
        ('chair.obj', 'chair.txt'),
        ('ship.obj', 'ship.txt'),
    ]
)
def test(input_file, solution_file):
    filename = str(Path(__file__).parent / input_file)
    bricks = Mesh2Brick()(filename)

    solution_filename = str(Path(__file__).parent / solution_file)

    with open(solution_filename) as f:
        solution = f.read()

    assert bricks.to_txt() == solution
