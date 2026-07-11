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
- Reads GameState, never writes it. `game_state` is an optional dependency
  purely for visualization: draw_player() reads `game_state.player_pos`
  to know where to draw, and does nothing else with it -- no move(),
  no tick(), no mutation of any kind. Gameplay logic (movement, win
  detection, timing) stays entirely in GameState/engine/game_state.py.
  This keeps the dependency arrow one-way: GameState -> MazeRenderer ->
  Renderer -> OpenGL. MazeRenderer depends on GameState's public
  read-only interface; GameState has no idea MazeRenderer exists.
"""

from typing import List, Optional, Tuple

from engine.game_state import GameState
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
        game_state: Optional[GameState] = None,
        player_color: Color = (0.2, 0.7, 1.0),
        player_size_ratio: float = 0.6,
        goal_color: Color = (0.3, 0.85, 0.4),
        goal_size_ratio: float = 0.6,
    ) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must be a non-empty 2D list of Cell objects")

        if not (0.0 < player_size_ratio <= 1.0):
            raise ValueError("player_size_ratio must be in the range (0.0, 1.0]")

        if not (0.0 < goal_size_ratio <= 1.0):
            raise ValueError("goal_size_ratio must be in the range (0.0, 1.0]")

        self.renderer: Renderer = renderer
        self.grid: List[List[Cell]] = grid

        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

        self.cell_size: int = cell_size
        self.margin_x: int = margin_x
        self.margin_y: int = margin_y

        self.wall_color: Color = wall_color
        self.wall_thickness: float = wall_thickness

        # Optional -- MazeRenderer works exactly as before (walls only) if
        # no GameState is supplied. Purely a read source for draw_player()/
        # draw_goal(); never mutated by this class. GameState remains the
        # single source of truth for both player_pos and goal_pos -- this
        # class never stores its own copy of either position, it re-reads
        # game_state on every draw call.
        self.game_state: Optional[GameState] = game_state
        self.player_color: Color = player_color
        self.player_size_ratio: float = player_size_ratio
        self.goal_color: Color = goal_color
        self.goal_size_ratio: float = goal_size_ratio

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

    def draw_goal(self) -> None:
        """
        Draw the goal as a filled square centered in GameState.goal_pos,
        via Renderer.draw_rect(). Reads game_state.goal_pos only -- does
        not store, cache, or mutate it. GameState remains the sole source
        of truth for where the goal is; this method just visualizes
        whatever it currently reports. No-op if no game_state was supplied.
        """
        if self.game_state is None:
            return

        row, col = self.game_state.goal_pos
        cx, cy = self.cell_center(row, col)

        size = self.cell_size * self.goal_size_ratio
        x = cx - size / 2
        y = cy - size / 2

        self.renderer.draw_rect(x, y, size, size, color=self.goal_color, filled=True)

    def draw_player(self) -> None:
        """
        Draw the player as a filled square centered in their current cell,
        via Renderer.draw_rect(). Reads game_state.player_pos only -- does
        not move the player, does not touch move_count/won/elapsed_time,
        and does not call any GameState methods. If no game_state was
        supplied to this MazeRenderer, this is a no-op.
        """
        if self.game_state is None:
            return

        row, col = self.game_state.player_pos
        cx, cy = self.cell_center(row, col)

        size = self.cell_size * self.player_size_ratio
        x = cx - size / 2
        y = cy - size / 2

        self.renderer.draw_rect(x, y, size, size, color=self.player_color, filled=True)

    def render(self) -> None:
        """
        Convenience: draw the maze, then the goal, then the player on top
        (if a game_state was supplied). Goal is drawn before the player so
        that if they ever occupy the same cell (i.e. right at the moment
        of winning), the player marker is the one left visible. Equivalent
        to calling draw_maze() + draw_goal() + draw_player() -- provided so
        callers (e.g. main.py's loop) have a single call site, without
        draw_maze()'s existing behavior changing for anyone still calling
        it directly.
        """
        self.draw_maze()
        self.draw_goal()
        self.draw_player()


if __name__ == "__main__":
    # Standalone smoke test: generate a small maze, wrap it in a GameState,
    # and visualize both the maze and the player -- with no other project
    # files (config.py, main.py, input_handler.py, etc.) involved. If this
    # shows a correctly-connected maze with a filled square sitting in the
    # top-left cell, wall+player rendering are both solid.
    from engine.game_state import GameState
    from maze.generator import MazeGenerator

    CELL_SIZE = 40
    MARGIN = 20

    maze = MazeGenerator(rows=15, cols=15, seed=42)
    maze.generate()
    game_state = GameState(maze)  # player starts at (0, 0) by default

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
            game_state=game_state,
        )

        while not renderer.should_close():
            renderer.begin_frame()
            maze_renderer.render()
            renderer.end_frame()
