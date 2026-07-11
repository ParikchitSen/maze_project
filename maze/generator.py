"""
generator.py

Defines MazeGenerator: builds a fully-connected 2D maze using iterative
(stack-based) depth-first search with backtracking.

Design notes:
- No recursion. Python's default recursion limit (~1000) would be blown by
  a 50x50 (2500-cell) maze doing recursive DFS, since a single winding path
  can visit every cell before backtracking. A plain list used as a stack
  gives identical DFS behavior without that ceiling.
- get_neighbors() is deliberately "dumb": it returns every in-bounds
  geometric neighbor as (direction, Cell) tuples, with no awareness of
  walls or `visited`. That's intentional — it's a pure grid-adjacency
  helper. generate() filters by `visited` itself, and a future BFS solver
  can filter by `has_wall(direction)` itself. Keeping this method
  unopinionated is what makes it safely reusable by both.
- All state (grid, rng, dimensions) lives on the instance. Nothing here
  touches module-level/global state, so multiple MazeGenerator instances
  (e.g. in tests) never interfere with each other.
"""

import random
from typing import List, Optional, Tuple

from maze.cell import Cell, DIRECTIONS, OPPOSITE

# Row/col offset for each direction. 'N' decreases row (moves up a line
# when printed), 'S' increases row, 'E' increases col, 'W' decreases col.
_DELTA = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


class MazeGenerator:
    """Generates a maze on a rows x cols grid using iterative DFS."""

    def __init__(self, rows: int, cols: int, seed: Optional[int] = None) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")

        self.rows: int = rows
        self.cols: int = cols

        # Dedicated RNG instance (not the global `random` module) so seeding
        # one MazeGenerator never affects another, or anything else in the
        # process that happens to use random. Makes reproducible mazes safe
        # to use in tests alongside unrelated randomness.
        self._rng: random.Random = random.Random(seed)

        # The grid is built here (not lazily in generate()) so callers can
        # inspect an empty, fully-walled grid immediately after construction
        # if they ever need to.
        self.grid: List[List[Cell]] = [
            [Cell(r, c) for c in range(cols)] for r in range(rows)
        ]

        self._generated: bool = False

    def in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is a valid grid coordinate."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_neighbors(self, cell: Cell) -> List[Tuple[str, Cell]]:
        """
        Return this cell's in-bounds geometric neighbors as
        (direction, neighbor_cell) tuples. Does NOT consider walls or
        `visited` — it's a plain adjacency lookup meant to be reused by
        generation, and later by the BFS solver, each of which will apply
        its own filtering.
        """
        neighbors: List[Tuple[str, Cell]] = []
        for direction in DIRECTIONS:
            dr, dc = _DELTA[direction]
            nr, nc = cell.row + dr, cell.col + dc
            if self.in_bounds(nr, nc):
                neighbors.append((direction, self.grid[nr][nc]))
        return neighbors

    def generate(self) -> List[List[Cell]]:
        """
        Carve the maze using iterative DFS with backtracking.

        Algorithm:
            1. Start at (0, 0), mark visited, push onto stack.
            2. While the stack is non-empty:
                a. Look at the cell on top of the stack.
                b. Find its unvisited neighbors.
                c. If there are any: pick one at random, remove the wall
                   between them, mark it visited, push it onto the stack.
                d. If there are none: pop the stack (backtrack).
            3. Stack empties exactly when every reachable cell has been
               visited. Since the grid is fully connected by construction
               (every cell has an unvisited-neighbor entry point until it's
               carved), this guarantees no isolated cells/islands.

        Returns self.grid for convenience, though callers can also just
        read self.grid directly after calling this.
        """
        start = self.grid[0][0]
        start.visited = True

        stack: List[Cell] = [start]

        while stack:
            current = stack[-1]  # peek, don't pop yet

            # Unvisited neighbors of the current cell.
            unvisited = [
                (direction, neighbor)
                for direction, neighbor in self.get_neighbors(current)
                if not neighbor.visited
            ]

            if unvisited:
                direction, chosen = self._rng.choice(unvisited)
                current.remove_wall(chosen, direction)
                chosen.visited = True
                stack.append(chosen)
            else:
                # Dead end: backtrack until we find a cell with an
                # unvisited neighbor, or the stack empties.
                stack.pop()

        self._generated = True
        return self.grid

    def print_ascii(self) -> None:
        """
        Print the maze to the console using +, -, | characters.

        Each cell is rendered as a 2-wide, 1-tall interior cell in a grid
        of corners. For every row of cells we print a "wall line" (top
        edge, showing N walls) followed by a "cell line" (showing W walls
        and floors), and a final wall line closes off the bottom (S walls
        of the last row).

        This runs entirely off Cell.has_wall(), so it doubles as a
        correctness check on remove_wall(): if remove_wall ever left one
        side of a shared wall up, this drawing would visibly show a
        one-sided gap or a missing corner-consistent wall.
        """
        if not self._generated:
            # Not a hard error — an unwalled grid is still valid to print
            # (it'll just show as fully walled-in), but flag it since it's
            # almost certainly not what the caller intended.
            print("[warning] print_ascii() called before generate()")

        # Top border: every cell's north wall on row 0.
        top = "+"
        for c in range(self.cols):
            top += "--+" if self.grid[0][c].has_wall("N") else "  +"
        print(top)

        for r in range(self.rows):
            # Cell line: west wall of col 0, then each cell's floor + east wall.
            row_line = "|" if self.grid[r][0].has_wall("W") else " "
            for c in range(self.cols):
                cell = self.grid[r][c]
                row_line += "  "
                row_line += "|" if cell.has_wall("E") else " "
            print(row_line)

            # Wall line below this row: each cell's south wall.
            wall_line = "+"
            for c in range(self.cols):
                cell = self.grid[r][c]
                wall_line += "--" if cell.has_wall("S") else "  "
                wall_line += "+"
            print(wall_line)


if __name__ == "__main__":
    maze = MazeGenerator(rows=10, cols=10, seed=42)
    maze.generate()
    maze.print_ascii()
