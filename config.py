"""
config.py

Centralized configuration constants shared across modules. Currently holds
rendering constants that need a single source of truth -- starting with
wall thickness, which main.py reads and passes into MazeRenderer.

Note: I (Claude) have no visibility into whatever you may already have in
your actual config.py -- if it already has content, merge this in rather
than overwriting.
"""

# Thickness, in pixels, of each maze wall when MazeRenderer draws it as a
# filled rectangle. Keep this meaningfully smaller than CELL_SIZE (see
# main.py) -- a thickness close to or larger than the cell size will start
# to visually swallow the playable interior of each cell.
WALL_THICKNESS: float = 4.0

# Floor tile drawn inside each cell, beneath the walls/goal/player. Color
# is a plain (r, g, b) tuple in the 0.0-1.0 range Renderer expects. Tiles
# are sized to fill each cell exactly (no gap/margin) so neighboring tiles
# touch seamlessly; MazeRenderer applies a small deterministic per-tile
# brightness variation on top of this base color for a subtle stone/
# concrete look.
FLOOR_COLOR: tuple = (0.13, 0.13, 0.17)
