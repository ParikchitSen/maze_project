"""
config.py

Centralized configuration constants shared across modules. Currently holds
rendering constants that need a single source of truth -- wall thickness
and appearance, and floor tile appearance -- which main.py reads and
passes into MazeRenderer.

Note: I (Claude) have no visibility into whatever you may already have in
your actual config.py -- if it already has content, merge this in rather
than overwriting.
"""

# Thickness, in pixels, of each maze wall when MazeRenderer draws it as a
# filled rectangle. 6-10 gives a "thick stone wall" feel; keep it
# meaningfully smaller than CELL_SIZE (see main.py) -- a thickness close
# to or larger than the cell size will start to swallow the playable
# interior of each cell.
WALL_THICKNESS: float = 8.0

# Warm stone/beige base color for wall fills (r, g, b in 0.0-1.0).
WALL_COLOR: tuple = (0.78, 0.66, 0.48)

# Darker outline drawn around every wall rectangle's true outer boundary,
# and its thickness in pixels.
WALL_OUTLINE_COLOR: tuple = (0.30, 0.22, 0.14)
WALL_OUTLINE_THICKNESS: float = 1.5

# Simple top-lit/bottom-shadowed bevel: fraction (0.0-1.0) by which the
# top edge strip is brightened and the bottom edge strip is darkened
# relative to WALL_COLOR, and the strip thickness in pixels.
WALL_HIGHLIGHT_STRENGTH: float = 0.18
WALL_SHADOW_STRENGTH: float = 0.18
WALL_LIGHTING_THICKNESS: float = 2.0

# Floor tile drawn inside each cell, beneath the walls/goal/player. Color
# is a plain (r, g, b) tuple in the 0.0-1.0 range Renderer expects. Tiles
# are sized to fill each cell exactly (no gap/margin) so neighboring tiles
# touch seamlessly; MazeRenderer applies a small deterministic per-tile
# brightness variation on top of this base color for a subtle stone/
# concrete look.
FLOOR_COLOR: tuple = (0.13, 0.13, 0.17)

# Minimum seconds between successive successful player moves while a
# direction key (arrow or WASD) is held down. 0.10-0.15 reads as one
# deliberate step per "tick" instead of a flood of moves every rendered
# frame; InputHandler is what actually enforces this, using
# Renderer.get_time() -- this is just the tunable value.
MOVE_REPEAT_DELAY: float = 0.12

# Fraction (0.0-1.0) of the window's tighter-fitting dimension the maze
# should occupy. 0.90-0.95 leaves a comfortable, roughly-equal margin on
# all sides while still filling most of the window. MazeRenderer
# recomputes cell_size/margins from this every frame against the
# Renderer's CURRENT width/height, so the maze stays correctly fitted and
# centered through window resizes and fullscreen toggles.
MAZE_FILL_FRACTION: float = 0.92
