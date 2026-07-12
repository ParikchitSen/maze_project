"""
engine/game_state.py

GameState: holds all gameplay state for a single maze run -- player
position, goal position, move count, elapsed time, and win state -- and
the movement API that mutates it.

Design notes (why it's built this way):
- Pure gameplay state, zero rendering. No OpenGL, no GLFW, no draw calls
  anywhere in this file. GameState only stores data (row/col ints, an int
  counter, a float, a bool) and exposes methods that update that data.
  A future renderer (e.g. a PlayerRenderer built on top of MazeRenderer)
  would READ from a GameState instance (its player_pos) to know where to
  draw a marker -- but GameState itself never draws anything or imports
  anything rendering-related.
- Movement reuses MazeGenerator.get_neighbors() instead of recomputing
  direction->coordinate deltas. get_neighbors() was explicitly built to be
  reused by future traversal code (its docstring says as much), so rather
  than duplicating the N/S/E/W-to-row/col-offset mapping here, GameState
  takes the MazeGenerator itself and asks it for the neighbor in a given
  direction. If that mapping ever changes, there is exactly one place
  (generator.py) that needs updating.
- Wall-checking is the single source of truth for "can I move here".
  Every movement method funnels through move(direction), which asks the
  player's CURRENT Cell whether the wall on that side is still up
  (has_wall()). There's no separate/duplicate notion of "reachability"
  anywhere else in this file.
- move_north/move_south/move_east/move_west are thin wrappers around a
  single internal move(direction) implementation, per the "generic move()
  API" ask -- so any future input-handling code can either call the named
  methods directly, or drive movement generically from e.g. a
  direction string coming out of input_handler.py, without GameState
  needing two parallel code paths.
- elapsed_time is a placeholder float, updated via tick(dt) rather than
  measuring wall-clock time itself. GameState shouldn't know or care
  whether "dt" comes from GLFW's timer, a fixed-step loop, or a test
  harness feeding it fake values -- that decision belongs to main.py.
"""

from typing import List, Optional, Tuple

from maze.cell import Cell
from maze.generator import MazeGenerator


class GameState:
    """
    Tracks player position, goal position, move count, elapsed time, and
    win state for a single maze run. Contains no rendering or OpenGL code.
    """

    def __init__(
        self,
        generator: MazeGenerator,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.generator: MazeGenerator = generator
        self.grid: List[List[Cell]] = generator.grid

        # Default start is the top-left corner, default goal is the
        # bottom-right corner -- the two cells DFS generation always
        # guarantees are reachable from one another, since every cell is
        # reachable by construction (see MazeGenerator.generate()).
        start_row, start_col = start if start is not None else (0, 0)
        goal_row, goal_col = (
            goal if goal is not None else (generator.rows - 1, generator.cols - 1)
        )

        self._validate_position(start_row, start_col, "start")
        self._validate_position(goal_row, goal_col, "goal")

        self.player_row: int = start_row
        self.player_col: int = start_col
        self.goal_row: int = goal_row
        self.goal_col: int = goal_col

        self.move_count: int = 0
        self.elapsed_time: float = 0.0
        self.won: bool = (self.player_row, self.player_col) == (goal_row, goal_col)

    # ------------------------------------------------------------------
    # Position access
    # ------------------------------------------------------------------

    @property
    def player_pos(self) -> Tuple[int, int]:
        """Current player position as (row, col)."""
        return self.player_row, self.player_col

    @property
    def goal_pos(self) -> Tuple[int, int]:
        """Goal position as (row, col)."""
        return self.goal_row, self.goal_col

    def current_cell(self) -> Cell:
        """The Cell the player currently occupies."""
        return self.grid[self.player_row][self.player_col]

    def _validate_position(self, row: int, col: int, label: str) -> None:
        if not (0 <= row < self.generator.rows and 0 <= col < self.generator.cols):
            raise ValueError(
                f"{label} position {(row, col)} is out of bounds for a "
                f"{self.generator.rows}x{self.generator.cols} maze"
            )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self, direction: str) -> bool:
        """
        Attempt to move the player one cell in `direction` ('N'/'S'/'E'/'W').

        Returns True if the move succeeded, False if a wall blocked it (or
        the game is already won). This is the single internal entry point
        every directional move method funnels through -- any future
        generic input handling can call this directly instead of picking a
        named method.
        """
        if self.won:
            return False

        current = self.current_cell()
        if current.has_wall(direction):
            return False

        neighbor = self._neighbor_in_direction(current, direction)
        if neighbor is None:
            # Defensive only: a passable wall should always have a
            # corresponding in-bounds neighbor, since remove_wall() is the
            # only way a wall comes down and it always operates between
            # two real, adjacent Cells.
            return False

        self.player_row, self.player_col = neighbor.row, neighbor.col
        self.move_count += 1
        self._check_win()
        return True

    def move_north(self) -> bool:
        return self.move("N")

    def move_south(self) -> bool:
        return self.move("S")

    def move_east(self) -> bool:
        return self.move("E")

    def move_west(self) -> bool:
        return self.move("W")

    def _neighbor_in_direction(self, cell: Cell, direction: str) -> Optional[Cell]:
        """
        Look up the geometric neighbor of `cell` in `direction`, via
        MazeGenerator.get_neighbors() -- reused rather than reimplemented
        so direction-to-coordinate logic lives in exactly one place.
        """
        for d, neighbor in self.generator.get_neighbors(cell):
            if d == direction:
                return neighbor
        return None

    # ------------------------------------------------------------------
    # Win detection
    # ------------------------------------------------------------------

    def _check_win(self) -> None:
        """
        Update `won` based on the current player position, and announce the
        win exactly once -- on the frame it FIRST becomes True, not on every
        subsequent move() call (movement is blocked after winning anyway,
        but this guards against ever calling _check_win() again while
        already won). This lives in GameState, not the renderer, because
        "you won" is a gameplay fact, not a visual -- the terminal message
        is just how that fact is surfaced right now, and could just as
        easily be swapped for something else later without renderer code
        changing at all.
        """
        was_won = self.won
        self.won = self.player_pos == self.goal_pos
        if self.won and not was_won:
            print("Maze Completed!")

    # ------------------------------------------------------------------
    # Time tracking
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """
        Advance the elapsed-time placeholder by `dt` seconds. No-ops once
        the game has been won, so elapsed_time freezes at the moment of
        completion instead of continuing to climb on the win screen --
        the same "stop progressing once won" rule move() already follows.
        main.py owns deciding where dt comes from (GLFW's clock, a fixed
        timestep, etc.) -- GameState just accumulates whatever it's given.
        """
        if self.won:
            return
        self.elapsed_time += dt

    def __repr__(self) -> str:
        return (
            f"GameState(player={self.player_pos}, goal={self.goal_pos}, "
            f"moves={self.move_count}, elapsed={self.elapsed_time:.2f}s, "
            f"won={self.won})"
        )
