"""
engine/maze_renderer.py

MazeRenderer: converts an already-generated maze grid into OpenGL line
segments, drawn entirely through the generic Renderer (engine/renderer.py).

Design notes (why it's built this way):
- Visualization only, no generation. This module takes a grid of Cell
  objects that has ALREADY been generated (by MazeGenerator elsewhere) and
  draws it. It never calls generate(), never touches remove_wall(), and
  never mutates a Cell. If you feed it a half-generated or freshly-created
  (all-walls-up) grid, it will happily draw that too -- it has no opinion
  on how the grid came to be.
- No direct OpenGL calls. Every pixel drawn goes through
  `self.renderer.draw_line(...)`. This keeps the "how do I talk to the
  GPU" knowledge confined to Renderer, and this file only ever reasons
  about maze geometry (rows/cols/cell_size/margins) and Cell wall state.
  If Renderer's backend ever changes (fixed-function -> shaders, or a
  different windowing library entirely), MazeRenderer would need zero
  changes as long as draw_line()'s signature stays the same.
- No duplicated generation logic. Row/col -> Cell lookups, wall checks
  (`cell.has_wall(...)`), and dimensions all come from the grid itself
  (via len()) -- this file does not reimplement anything from
  maze/generator.py or maze/cell.py, it only reads from Cell's public
  interface (`has_wall`).
- One wall, one line. Every standing wall in the maze is drawn exactly
  once: each cell draws its own N and W walls; only cells in the last row
  / last column additionally draw their S / E walls. Since remove_wall()
  guarantees both sides of a shared wall always agree, this covers every
  wall in the maze with no duplicate (overlapping) line draws.
- Built for reuse by later layers. `cell_to_pixel()`, `cell_rect()`, and
  `cell_center()` are public on purpose: a future PlayerRenderer,
  GoalRenderer, or SolverPathRenderer can import MazeRenderer, reuse these
  same pixel-mapping helpers, and draw on top of an already-drawn maze
  without recomputing cell_size/margin math themselves or risking it
  drifting out of sync with how the maze itself is drawn.
"""

from typing import List, Tuple

from engine.renderer import Color, Renderer
from maze.cell import Cell


class MazeRenderer:
    """
    Draws an already-generated maze grid using a Renderer instance.
    Holds no maze-generation logic -- purely a grid-of-Cells -> lines
    translator.
    """

    def __init__(
        self,
        renderer: Renderer,
        grid: List[List[Cell]],
        cell_size: int = 40,
        margin_x: int = 20,
        margin_y: int = 20,
        wall_color: Color = (0.9, 0.9, 0.9),
        wall_thickness: float = 2.0,
    ) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must be a non-empty 2D list of Cell objects")

        self.renderer: Renderer = renderer
        self.grid: List[List[Cell]] = grid

        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

        self.cell_size: int = cell_size
        self.margin_x: int = margin_x
        self.margin_y: int = margin_y

        self.wall_color: Color = wall_color
        self.wall_thickness: float = wall_thickness

    # ------------------------------------------------------------------
    # Pixel-mapping helpers (public, reusable by future overlay renderers)
    # ------------------------------------------------------------------

    def cell_to_pixel(self, row: int, col: int) -> Tuple[float, float]:
        """
        Return the pixel coordinates of this cell's TOP-LEFT corner, in the
        same top-left-origin, y-down coordinate system Renderer draws in.
        """
        x = self.margin_x + col * self.cell_size
        y = self.margin_y + row * self.cell_size
        return x, y

    def cell_rect(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """
        Return (x, y, w, h) for this cell's full square, suitable for
        Renderer.draw_rect(). Useful later for highlighting a cell (e.g. the
        goal) or drawing a filled player marker inset within it.
        """
        x, y = self.cell_to_pixel(row, col)
        return x, y, self.cell_size, self.cell_size

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """
        Return the pixel coordinates of this cell's center. Useful later
        for drawing a player/goal marker or a solver-path line that should
        pass through the middle of each cell rather than along its edges.
        """
        x, y = self.cell_to_pixel(row, col)
        half = self.cell_size / 2
        return x + half, y + half

    def total_pixel_size(self) -> Tuple[int, int]:
        """
        Return the (width, height) in pixels needed to display this maze
        with the configured cell_size and margins. Handy for sizing the
        Renderer's window to exactly fit the maze instead of guessing.
        """
        return MazeRenderer.compute_window_size(
            self.rows, self.cols, self.cell_size, self.margin_x, self.margin_y
        )

    @staticmethod
    def compute_window_size(
        rows: int, cols: int, cell_size: int, margin_x: int, margin_y: int
    ) -> Tuple[int, int]:
        """
        Compute the (width, height) in pixels a maze of this shape would
        need, WITHOUT requiring a Renderer or MazeRenderer instance to
        exist yet. This lets callers size their Renderer's window correctly
        on the very first call, instead of picking an arbitrary width/height
        and hoping it fits (see the __main__ block below).
        """
        width = margin_x * 2 + cols * cell_size
        height = margin_y * 2 + rows * cell_size
        return width, height

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_maze(self) -> None:
        """
        Draw every standing wall in the maze as a line segment, via
        Renderer.draw_line(). Call this once per frame, between
        Renderer.begin_frame() and Renderer.end_frame().
        """
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.grid[row][col]
                x, y = self.cell_to_pixel(row, col)
                size = self.cell_size

                # Every cell draws its own N and W walls unconditionally.
                if cell.has_wall("N"):
                    self.renderer.draw_line(
                        x, y, x + size, y,
                        color=self.wall_color, line_width=self.wall_thickness,
                    )
                if cell.has_wall("W"):
                    self.renderer.draw_line(
                        x, y, x, y + size,
                        color=self.wall_color, line_width=self.wall_thickness,
                    )

                # S and E walls are only drawn from the last row / last
                # column -- every other S/E wall is already covered by the
                # N/W wall of the neighboring cell (remove_wall guarantees
                # both sides always agree, so this never misses a wall).
                if row == self.rows - 1 and cell.has_wall("S"):
                    self.renderer.draw_line(
                        x, y + size, x + size, y + size,
                        color=self.wall_color, line_width=self.wall_thickness,
                    )
                if col == self.cols - 1 and cell.has_wall("E"):
                    self.renderer.draw_line(
                        x + size, y, x + size, y + size,
                        color=self.wall_color, line_width=self.wall_thickness,
                    )


if __name__ == "__main__":
    # Standalone smoke test: generate a small maze and visualize it, with
    # no other project files (config.py, main.py, solver.py, etc.) involved.
    # If this shows a correctly-connected maze with no stray gaps or double
    # walls, MazeRenderer is solid.
    from maze.generator import MazeGenerator

    CELL_SIZE = 40
    MARGIN = 20

    maze = MazeGenerator(rows=15, cols=15, seed=42)
    maze.generate()

    # Compute the exact window size this maze needs before the Renderer
    # (and its window) is even created -- no dummy/placeholder instance
    # required.
    width, height = MazeRenderer.compute_window_size(
        rows=maze.rows, cols=maze.cols, cell_size=CELL_SIZE, margin_x=MARGIN, margin_y=MARGIN
    )

    with Renderer(width=width, height=height, title="MazeRenderer smoke test") as renderer:
        maze_renderer = MazeRenderer(
            renderer=renderer,
            grid=maze.grid,
            cell_size=CELL_SIZE,
            margin_x=MARGIN,
            margin_y=MARGIN,
        )

        while not renderer.should_close():
            renderer.begin_frame()
            maze_renderer.draw_maze()
            renderer.end_frame()
