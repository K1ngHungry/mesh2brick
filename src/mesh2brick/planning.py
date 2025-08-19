from mesh2brick.data.brick_library import brick_library
from mesh2brick.data.brick_structure import Brick


def brick_priority(brick: Brick):
    return brick.x, brick.y


def plan_robotic_operation(brick_graph: dict[int, list[Brick]]):
    directed_brick_graph = dict()
    seq_num = 1
    bricks_track = dict()
    key_int = []
    for key in brick_graph.keys():
        key_int.append(int(key))
    sorted_key_int = sorted(key_int)
    for key in sorted_key_int:
        bricks = brick_graph[key]
        brick_layer_set = []
        for i in range(len(bricks)):
            brick = bricks[i]
            brick_layer_set.append(brick)
        brick_layer_set.sort(reverse=True, key=brick_priority)
        for brick in brick_layer_set:
            brick_id = str(brick.brick_id)
            if brick_id in bricks_track.keys():
                if bricks_track[brick_id] < brick_library[brick_id]['inventory']:
                    bricks_track[brick_id] += 1
                else:
                    continue
            else:
                bricks_track[brick_id] = 1

            # Write brick entry: (id, x, y, z, orientation)
            directed_brick_graph[str(seq_num)] = {'brick_id': brick.brick_id,
                                                 'x': brick.x,
                                                 'y': brick.y,
                                                 'z': brick.z,
                                                 'ori': brick.ori}
            seq_num += 1
    return directed_brick_graph
