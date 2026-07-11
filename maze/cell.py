"""
cell.py

Defines the Cell class: a single grid cell in the maze.

Design notes (why it's built this way):
- `walls` is a dict keyed by 'N', 'S', 'E', 'W' so directions are explicit
  everywhere in the codebase instead of magic indices (0/1/2/3). This keeps
  generator.py, and later solver.py / renderer.py, readable.
- `remove_wall` is the ONLY way walls should ever be knocked down. It takes
  the neighboring Cell and the direction *from self to that neighbor*, and
  updates both cells atomically. This guarantees the grid can never end up
  in an inconsistent state (e.g. this cell thinks the wall is gone but the
  neighbor still thinks it's up) because there is no other code path that
  touches `walls` directly.
- `visited` is generation-only bookkeeping. It's kept on the Cell (rather
  than a separate set in the generator) because it's simple, and because a
  future solver can trivially reset it if it ever wants to reuse the field
  for its own traversal (see reset_visited below).
"""

from typing import Dict


# Maps each direction to its opposite. Shared as a module-level constant
# so generator.py can reuse it too (e.g. get_neighbors) without duplicating
# the mapping or risking it drifting out of sync with Cell's internal logic.
OPPOSITE: Dict[str, str] = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
}

# Canonical direction order, used by __repr__ and print_ascii for stable,
# deterministic output.
DIRECTIONS = ("N", "S", "E", "W")


class Cell:
    """A single cell in the maze grid, tracking its own walls."""

    def __init__(self, row: int, col: int) -> None:
        self.row: int = row
        self.col: int = col

        # All four walls start up. This dict is the single source of truth
        # for a cell's connectivity to its neighbors.
        self.walls: Dict[str, bool] = {"N": True, "S": True, "E": True, "W": True}

        # Generation-only flag. Not meaningful once generate() has finished,
        # except as a record of the fact that every reachable cell was
        # visited (i.e. the maze is fully connected).
        self.visited: bool = False

    def has_wall(self, direction: str) -> bool:
        """Return True if the wall on the given side ('N'/'S'/'E'/'W') is up."""
        return self.walls[direction]

    def remove_wall(self, other_cell: "Cell", direction: str) -> None:
        """
        Remove the wall between this cell and `other_cell`.

        `direction` is the direction from THIS cell to `other_cell`
        (e.g. direction='N' means other_cell is north of self). This knocks
        down self's wall on that side AND other_cell's wall on the opposite
        side in one call, so callers never have to remember to update both
        cells themselves.
        """
        self.walls[direction] = False
        other_cell.walls[OPPOSITE[direction]] = False

    def reset_visited(self) -> None:
        """
        Reset the generation-only `visited` flag.

        Provided so a future BFS/DFS solver can reuse this same field for
        its own traversal bookkeeping without generator.py and solver.py
        stepping on each other silently.
        """
        self.visited = False

    def __repr__(self) -> str:
        up = "".join(d for d in DIRECTIONS if self.walls[d])
        return f"Cell(row={self.row}, col={self.col}, walls_up={up!r})"
