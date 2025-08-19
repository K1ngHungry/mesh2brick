# mesh2brick

`mesh2brick` is a utility package for converting an input mesh file into a brick structure in JSON, text, or LDraw
format.

## Usage

### Standalone project

To use as a standalone project, first install the package manager [uv](https://docs.astral.sh/uv/). Then, clone the
repository:

```bash
git clone https://github.com/AvaLovelace1/BrickGPT.git
cd BrickGPT/src/mesh2brick
```

Finally, run the `mesh2brick` script with the following command. `uv` automatically installs all dependencies upon
running.

```
uv run mesh2brick [INPUT_MESH_FILE] [OUTPUT_BRICK_FILE]
```

The `[INPUT_MESH_FILE]` can be any mesh format (`.obj`, `.glb`, etc.).

The `[OUTPUT_BRICK_FILE]` should end in `.json`, `.txt`, or `.ldr` to specify one of the following three output formats:

- **JSON.** The output JSON is a list of bricks. Each brick has the following keys:
    - `brick_id`: A number indicating the brick type. See `brick_library.json` for a list of brick types.
    - `x`, `y`, `z`: The coordinates of the brick in 3D space.
    - `ori`: 0 or 1, indicating the orientation of the brick.
- **Text.** The output text file is a list of bricks, one per line, with the following format: `hxw (x,y,z)`, where `h` is
  the length of the brick along the *x*-axis, `w` is the length of the brick along the *y*-axis, and `(x,y,z)` are the
  coordinates of the brick in 3D space.
- **LDraw.** The output LDraw file can be used directly with LDraw-compatible software to visualize the brick structure.

Run `uv run mesh2brick --help` to see all available options.

### Python package

To install `mesh2brick` as a package in an existing project, run:

```bash
pip install "mesh2brick @ git+https://github.com/AvaLovelace1/BrickGPT.git/#subdirectory=src/mesh2brick"
```

if using `pip`, or

```bash
uv add "mesh2brick @ git+https://github.com/AvaLovelace1/BrickGPT.git/#subdirectory=src/mesh2brick"
```

if using `uv`.

Then, you can use the `mesh2brick` module in your Python code:

```python
from mesh2brick import Mesh2Brick

input_file = "path/to/your/input_mesh.obj"
mesh2brick = Mesh2Brick()
output_bricks = mesh2brick(input_file)
```