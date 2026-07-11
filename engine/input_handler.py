"""
engine/input_handler.py

InputHandler: reads keyboard state each frame (via Renderer.is_key_pressed,
which wraps GLFW) and translates key presses into GameState movement calls.

Design notes (why it's built this way):
- GLFW input, accessed through Renderer, not imported directly. Key state
  is read through Renderer.is_key_pressed("UP") etc. -- plain strings, not
  raw GLFW key constants. Renderer's docstring establishes it as the only
  file that imports glfw/OpenGL; input polling is inherently tied to a
  specific GLFW window, which Renderer already owns, so InputHandler asks
  Renderer rather than importing glfw a second time. This still satisfies
  "use GLFW keyboard input" -- it's GLFW underneath -- while keeping GLFW
  encapsulated in one place. (If you'd rather this file call
  glfw.get_key(renderer.window, ...) directly, that's a one-file change --
  say the word and I'll flip it.)
- Independent of rendering. This file draws nothing -- no draw_line/
  draw_rect/begin_frame/end_frame calls anywhere. It only reads input and
  calls GameState methods. Rendering happens elsewhere in the frame loop
  (main.py), not here.
- Does not touch maze generation. No import of MazeGenerator/Cell, no wall
  logic of its own -- movement legality is entirely GameState's job
  (via has_wall on the player's current Cell). InputHandler only decides
  WHEN to attempt a move, never whether it's allowed.
- Only mutates GameState through its public move API (move_north(),
  move_south(), move_east(), move_west()) -- never player_row/player_col
  or any other internal directly.
- Edge-triggered by default (no key-repeat). `_just_pressed()` is the one
  primitive behind this: it compares "is the key down now" against "was it
  down last frame" and only reports True on the up->down transition. Without
  this, holding a direction key would call gs.move() every single frame
  (dozens of times a second) and the player would rocket across the maze
  on one held keypress instead of moving one cell per press. This is a
  general-purpose primitive, not movement-specific -- see below.
- Built to extend to regenerate/pause. `_just_pressed()` is keyed
  generically by key name (not pre-scoped to movement keys), so restart
  ('R') is already wired up via `is_restart_requested()`, and regenerate
  maze ('N') / pause ('P') can be added the same way later -- the
  edge-detection bookkeeping is already generic enough to cover them
  without changes.
"""

from typing import Dict

from engine.game_state import GameState
from engine.renderer import Renderer

# Each direction is bound to two keys (arrow key + WASD), since both are a
# common expectation and supporting both costs nothing.
MOVE_KEY_BINDINGS: Dict[str, str] = {
    "UP": "N",
    "W": "N",
    "DOWN": "S",
    "S": "S",
    "RIGHT": "E",
    "D": "E",
    "LEFT": "W",
    "A": "W",
}


class InputHandler:
    """
    Polls keyboard state once per frame and drives a GameState's movement
    accordingly. Holds no rendering or maze-generation logic of its own.
    """

    def __init__(self, renderer: Renderer, game_state: GameState) -> None:
        self.renderer: Renderer = renderer
        self.game_state: GameState = game_state

        # Tracks whether EACH key we've ever checked was down on the
        # previous frame, keyed by key name -- not pre-scoped to movement
        # keys, so future action keys (R/N/P) can reuse this same dict via
        # _just_pressed() without a second bookkeeping structure.
        self._was_pressed: Dict[str, bool] = {}

    def _just_pressed(self, key_name: str) -> bool:
        """
        Return True only on the frame `key_name` transitions from up to
        down (edge-triggered). Returns False on every subsequent frame the
        key stays held, so one press = one action -- no repeat-fire while
        held. This is the general primitive both movement and any future
        action keys (restart/regenerate/pause) are built on.
        """
        is_down = self.renderer.is_key_pressed(key_name)
        was_down = self._was_pressed.get(key_name, False)
        self._was_pressed[key_name] = is_down
        return is_down and not was_down

    def process_input(self) -> None:
        """
        Poll all bound keys once per frame and move the player on any
        newly-pressed movement key. Call this once per frame, before
        rendering, so the render reflects the result of any move made this
        frame. Safe to call even after the game is won -- GameState.move()
        is already a no-op once `won` is True, so no win-state check is
        needed here.
        """
        for key_name, direction in MOVE_KEY_BINDINGS.items():
            if self._just_pressed(key_name):
                self.game_state.move(direction)

        # Reserved for a future milestone -- regenerate maze ('N') and
        # pause ('P') will each hook in here as additional
        # `if self._just_pressed("N"): ...` checks, using the same
        # edge-triggered primitive above.

    def is_restart_requested(self) -> bool:
        """
        Return True exactly once, on the frame 'R' is newly pressed.

        InputHandler only REPORTS the request here -- it does not rebuild
        MazeGenerator/GameState/MazeRenderer itself. It has no reference to
        MazeGenerator and no authority to replace GameState's grid; that's
        orchestration, and belongs in main.py, which is what actually owns
        those objects' lifetimes. Kept as a separate method from
        process_input() (rather than folded into it) so main.py can check
        it explicitly and decide what "restart" means at the orchestration
        level.
        """
        return self._just_pressed("R")
