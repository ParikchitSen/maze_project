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
