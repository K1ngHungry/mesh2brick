import json
from pathlib import Path

with open(Path(__file__).parent / 'brick_library.json') as f:
    brick_library = json.load(f)  # Maps brick ID to brick properties

max_brick_dimension = max(max(properties['length'], properties['width']) for properties in brick_library.values())


def _make_dimensions_to_brick_id_dict() -> dict:
    result = {}
    for brick_id, properties in brick_library.items():
        key = (properties['length'], properties['width'], properties['height'])
        if key not in result.keys():
            result[key] = int(brick_id)
    return result


_dimensions_to_brick_id_dict = _make_dimensions_to_brick_id_dict()


def dimensions_to_brick_id(l: int, w: int, h: int):
    if l > w:
        l, w = w, l
    try:
        return _dimensions_to_brick_id_dict[(l, w, h)]
    except KeyError:
        raise ValueError(f'No brick ID for brick of dimensions: {l}x{w}x{h}')


def brick_id_to_dimensions(brick_id: int) -> (int, int, int):
    return brick_library[str(brick_id)]['length'], brick_library[str(brick_id)]['width'], brick_library[str(brick_id)]['height']


def brick_id_to_part_id(brick_id: int) -> str:
    """
    Returns the part ID of the given brick, which is the ID of the brick model used in LDraw files.
    """
    return brick_library[str(brick_id)]['partID']


def part_id_to_brick_id(part_id: str) -> int:
    """
    Returns the brick ID of the given part ID, which is the ID of the brick used in the brick library.
    """
    for brick_id, properties in brick_library.items():
        if properties['partID'] == part_id:
            return int(brick_id)
    raise ValueError(f'No brick ID for part ID: {part_id}')
