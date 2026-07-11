"""
main.py

Single entry point for the application. Wires together the maze-generation
layer and the rendering layer, and runs the main loop.

Design notes:
- Orchestration only. Every line here is "call module X, then call module
  Y" -- there is no maze-generation logic (that's MazeGenerator/Cell), no
  OpenGL/GLFW logic (that's Renderer), and no wall-to-pixel translation
  (that's MazeRenderer). If you find yourself wanting to add a for-loop
  over cells or a glBegin/glEnd call here, it belongs in one of those
  modules instead, not here.
- Config lives as module-level constants for now (rows/cols/cell_size/
  margins/seed). Easy to swap for config.py-driven values, CLI args, or a
  settings menu later without touching the loop's structure.
- The main loop is intentionally minimal: begin_frame -> render -> end_frame,
  repeated until the window wants to close. This is the shape that a future
  player-movement / game-state milestone will extend -- e.g. adding
  "process_input()" and "update_game_state()" calls between begin_frame()
  and the render calls, and a "draw_player()" call alongside draw_maze().
  The loop's skeleton is deliberately left simple so those additions are
  small, localized edits rather than a restructure.
"""

from engine.maze_renderer import MazeRenderer
from engine.renderer import Renderer
from maze.generator import MazeGenerator

# Maze configuration. Swap these for config.py / CLI args later without
# touching anything below.
ROWS = 15
COLS = 15
SEED = 42

# Rendering configuration.
CELL_SIZE = 40
MARGIN_X = 20
MARGIN_Y = 20
WINDOW_TITLE = "Maze"


def main() -> None:
    # 1. Generate the maze (pure data, no rendering involved yet).
    maze = MazeGenerator(rows=ROWS, cols=COLS, seed=SEED)
    maze.generate()

    # 2. Size the window to exactly fit this maze, then open it.
    width, height = MazeRenderer.compute_window_size(
        rows=maze.rows, cols=maze.cols,
        cell_size=CELL_SIZE, margin_x=MARGIN_X, margin_y=MARGIN_Y,
    )

    with Renderer(width=width, height=height, title=WINDOW_TITLE) as renderer:
        # 3. Wrap the generated grid for drawing.
        maze_renderer = MazeRenderer(
            renderer=renderer,
            grid=maze.grid,
            cell_size=CELL_SIZE,
            margin_x=MARGIN_X,
            margin_y=MARGIN_Y,
        )

        # 4. Main loop: begin frame -> render -> end frame, until closed.
        while not renderer.should_close():
            renderer.begin_frame()
            maze_renderer.draw_maze()
            renderer.end_frame()


if __name__ == "__main__":
    main()
