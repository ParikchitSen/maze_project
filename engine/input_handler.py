"""
engine/input_handler.py

InputHandler: reads keyboard state each frame (via Renderer.is_key_pressed,
which wraps glfw.get_key()) and translates that into GameState movement
calls and restart requests.

Design notes (why it's built this way):
- GLFW input, accessed through Renderer, not imported directly. Key state
  is read through Renderer.is_key_pressed("UP") etc. -- plain strings, not
  raw GLFW key constants. Renderer's docstring establishes it as the only
  file that imports glfw/OpenGL; input polling is inherently tied to a
  specific GLFW window, which Renderer already owns, so InputHandler asks
  Renderer rather than importing glfw a second time. Under the hood this
  IS glfw.get_key() polled every frame -- there is no GLFW key-callback/
  event-based path anywhere in this project, only polling.
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
- Movement is POLLED and TIME-GATED, not edge-triggered. Every frame,
  process_input() checks whether a bound key is currently held down
  (glfw.get_key() state, via Renderer) -- not whether it was JUST pressed.
  As long as a direction key is held, GameState.move() is attempted again
  and again, every frame, for as long as the key stays down. What keeps
  this from moving the player 60+ cells a second is move_repeat_delay: a
  minimum number of seconds that must pass since the last SUCCESSFUL move
  before another one is attempted. This timer (_last_move_time) is the
  "timing state" this module owns -- see below for why it lives here.
- Timing state lives in InputHandler, not GameState or Renderer.
  _last_move_time and move_repeat_delay are input/timing concerns, not
  gameplay state (GameState doesn't need to know WHY a move happened, just
  THAT it did) and not rendering state (Renderer has no opinion on how
  often the player should be allowed to move). InputHandler is the layer
  that turns "raw input over time" into "discrete move attempts," so the
  timer that paces those attempts belongs here, in one place, rather than
  a frame counter in main.py or a hidden clock inside GameState.
- Only the timer, not GLFW, is store here. InputHandler still doesn't
  import glfw or read the system clock directly -- it asks Renderer for
  the current time (Renderer.get_time(), which already existed for the
  elapsed-time HUD) the same way it asks Renderer for key state. Renderer
  remains the only file touching GLFW; GameState remains fully unaware
  that timing is involved in movement at all.
- Restart stays edge-triggered. 'R' should fire once per press, not repeat
  every frame it's held, so is_restart_requested() keeps using the
  original up->down edge-detection primitive (_just_pressed) -- a
  completely separate mechanism from the new movement polling, since the
  two keys need different behavior (repeat vs. one-shot).
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
    accordingly, at a fixed repeat interval while a direction key is held.
    Holds no rendering or maze-generation logic of its own.
    """

    def __init__(
        self,
        renderer: Renderer,
        game_state: GameState,
        move_repeat_delay: float = 0.12,
    ) -> None:
        self.renderer: Renderer = renderer
        self.game_state: GameState = game_state

        # Minimum seconds between successive successful moves while a
        # direction key is held (~100-150ms feels like one deliberate step
        # per "tick" rather than a flood of moves every rendered frame).
        # Configurable per-instance rather than hardcoded so main.py can
        # source the actual value from config.py, same as every other
        # tunable in this project.
        self.move_repeat_delay: float = move_repeat_delay

        # Timestamp (Renderer.get_time()) of the last SUCCESSFUL move.
        # Starts at -infinity so the very first press of a direction key
        # moves immediately, with no artificial startup delay.
        self._last_move_time: float = float("-inf")

        # Tracks whether EACH key we've ever checked was down on the
        # previous frame, keyed by key name -- used ONLY by the
        # edge-triggered primitive below (currently just restart). Not
        # involved in movement anymore, which is polled/time-gated instead.
        self._was_pressed: Dict[str, bool] = {}

    def _just_pressed(self, key_name: str) -> bool:
        """
        Return True only on the frame `key_name` transitions from up to
        down (edge-triggered). Returns False on every subsequent frame the
        key stays held. Used for one-shot actions like restart -- NOT used
        for movement anymore (see process_input()).
        """
        is_down = self.renderer.is_key_pressed(key_name)
        was_down = self._was_pressed.get(key_name, False)
        self._was_pressed[key_name] = is_down
        return is_down and not was_down

    def process_input(self) -> None:
        """
        Poll every bound movement key's CURRENT state (held or not) every
        frame, and attempt a move once move_repeat_delay seconds have
        passed since the last successful one. Call this once per frame,
        before rendering, so the render reflects the result of any move
        made this frame.

        Only one direction is attempted per frame (the first bound key
        found held, in MOVE_KEY_BINDINGS order) -- if two opposite keys are
        held at once, this avoids attempting both in the same frame. The
        timer only advances on a SUCCESSFUL move (GameState.move() returns
        True); a move blocked by a wall doesn't burn the cooldown, so
        switching to a valid direction while pressed against a wall isn't
        penalized with extra delay.

        Safe to call even after the game is won -- GameState.move() is
        already a no-op once `won` is True, so no win-state check is
        needed here.
        """
        now = self.renderer.get_time()

        for key_name, direction in MOVE_KEY_BINDINGS.items():
            if not self.renderer.is_key_pressed(key_name):
                continue
            if now - self._last_move_time < self.move_repeat_delay:
                break  # held, but not time for another step yet
            if self.game_state.move(direction):
                self._last_move_time = now
            break  # only ever act on one direction per frame

        # Reserved for a future milestone -- regenerate maze ('N') and
        # pause ('P') will each hook in here as additional
        # `if self._just_pressed("N"): ...` checks, using the
        # edge-triggered primitive above (these are one-shot actions, not
        # repeat-while-held, so they stay on _just_pressed rather than the
        # new polling/cooldown mechanism).

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
        level. Still edge-triggered (_just_pressed), unaffected by the
        movement polling change above -- restart is a one-shot action.
        """
        return self._just_pressed("R")
