import argparse
import json

from mesh2brick.mesh2brick import Mesh2Brick


def main():
    args = parse_args()

    mesh2brick = Mesh2Brick(world_dim=(args.world_dim, args.world_dim, args.world_dim), max_failures=args.max_failures)
    bricks = mesh2brick(args.input_file, x_rotation=args.x_rotation)

    if args.output_file.endswith('.json'):
        with open(args.output_file, 'w') as f:
            json.dump(bricks.to_json(), f)
    elif args.output_file.endswith('.txt'):
        with open(args.output_file, 'w') as f:
            f.write(bricks.to_txt())
    elif args.output_file.endswith('.ldr'):
        with open(args.output_file, 'w') as f:
            f.write(bricks.to_ldr())
    else:
        raise ValueError(f'Output filename must end in .json, .txt, or .ldr: {args.output_file}')


def parse_args():
    parser = argparse.ArgumentParser(
        prog='mesh2brick',
        description='Convert an input mesh to a brick structure.',
    )
    parser.add_argument('input_file', type=str, help='Filename of the input mesh.')
    parser.add_argument('output_file', type=str,
                        help='Filename of the output brick structure. Must end in .json, .txt, or .ldr')
    parser.add_argument('--world_dim', type=int, default=20,
                        help='World dimension. The output brick structure will fit within a cube of this size.')
    parser.add_argument('--max_failures', type=int, default=10,
                        help='Maximum number of failed re-merge attempts in the mesh2brick algorithm before timing out.')
    parser.add_argument('--x_rotation', type=int, default=90,
                        help='Rotation of the input mesh around the x-axis in degrees.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
