"""
main.py

Single entry point for the application. Wires together maze generation,
gameplay state, input handling, and rendering, and runs the main loop.

Design notes:
- Orchestration only. Every line here is "call module X, then call module
  Y" -- there is no maze-generation logic (that's MazeGenerator/Cell), no
  movement/win logic (that's GameState), no key-code handling (that's
  Renderer/InputHandler), and no wall/player/goal-to-pixel translation
  (that's MazeRenderer). If you find yourself wanting to add an
  if-wall-blocked check or a glBegin/glEnd call here, it belongs in one of
  those modules instead, not here.
- Config lives as module-level constants for now (rows/cols/window size).
  Easy to swap for config.py-driven values, CLI args, or a settings menu
  later without touching the loop's structure. SEED is intentionally
  None -- a fresh random maze every run, since a maze that's identical
  every time gets boring fast. Pass a fixed int here temporarily if you
  ever need a reproducible maze for debugging.
- No text-rendering library yet. Move count and elapsed time are shown in
  the WINDOW TITLE (via Renderer.set_title()) instead of drawn on-screen --
  the OS draws the title bar for free, so this is a simple way to surface
  live info without pulling in a font-rendering dependency.
- GameState stays independent of GLFW timing. main.py computes dt via
  Renderer.get_time() (the only file that touches GLFW) and hands GameState
  a plain float via tick(dt) -- GameState has no idea where dt came from.
- Restart ('R') rebuilds MazeGenerator + GameState + MazeRenderer by
  calling the SAME local `build_session()` function used for the initial
  setup -- reusing the existing classes rather than writing separate
  "reset" logic for each piece of state. No global variables: maze,
  game_state, and maze_renderer are plain local variables in main(),
  reassigned in place when a restart happens. Renderer and InputHandler
  are NOT rebuilt on restart: the window doesn't need to be recreated
  (dimensions don't change), and InputHandler's key state needs to persist
  across the restart -- if InputHandler were recreated, and the 'R' key
  was still physically held down at that instant, a fresh instance would
  immediately see "just pressed" again and could re-trigger a restart loop
  while the key is held. Reusing the same InputHandler (and simply
  reassigning its .game_state to the new one) avoids that.
- Fullscreen ('F11') is a one-line call. main.py never touches GLFW,
  never remembers window geometry, and never recomputes cell_size/margins
  itself -- it just calls renderer.toggle_fullscreen() when InputHandler
  reports F11 was pressed. Renderer owns everything about how the toggle
  actually happens (glfw.set_window_monitor, preserving the GL context,
  restoring windowed position/size); MazeRenderer owns re-fitting the maze
  to whatever size the window ends up being, every frame, on its own --
  main.py doesn't need to know a resize happened at all.
"""

from engine.game_state import GameState
from engine.input_handler import InputHandler
from engine.maze_renderer import MazeRenderer
from engine.renderer import Renderer
from maze.generator import MazeGenerator
from config import (
    WALL_THICKNESS,
    WALL_COLOR,
    WALL_OUTLINE_COLOR,
    WALL_OUTLINE_THICKNESS,
    WALL_HIGHLIGHT_STRENGTH,
    WALL_SHADOW_STRENGTH,
    WALL_LIGHTING_THICKNESS,
    FLOOR_COLOR,
    MOVE_REPEAT_DELAY,
    MAZE_FILL_FRACTION,
)

# Maze configuration. Swap these for config.py / CLI args later without
# touching anything below. SEED=None means a new layout every time a maze
# is generated -- both on startup and every time 'R' is pressed.
ROWS = 15
COLS = 15
SEED = None

# Rendering configuration. WINDOW_WIDTH/HEIGHT are only the STARTING
# windowed-mode size -- MazeRenderer no longer takes a fixed cell_size or
# margins; it derives both every frame from Renderer's current width/height
# (see MazeRenderer._update_layout()), so the maze automatically re-fits
# itself on any resize or fullscreen toggle without main.py doing anything.
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_TITLE = "Maze"


def main() -> None:
    def build_session() -> tuple:
        """
        Build a brand-new (maze, game_state) pair -- a fresh MazeGenerator
        run with a new random layout, and a fresh GameState pointed at it
        (player back at the start, move_count/elapsed_time/won all reset
        to their defaults simply by virtue of being a new GameState
        instance). Used both for the initial maze and for every restart,
        so there's exactly one place that knows how to set up a session.
        """
        new_maze = MazeGenerator(rows=ROWS, cols=COLS, seed=SEED)
        new_maze.generate()
        new_game_state = GameState(new_maze)
        return new_maze, new_game_state

    # 1. Build the initial maze + gameplay state.
    maze, game_state = build_session()

    # 2. Open the window at a starting windowed-mode size. This is ONLY
    # the initial size -- it has no special relationship to cell_size or
    # the maze's dimensions anymore (see MazeRenderer._update_layout()),
    # so there's nothing to compute here, just a fixed starting point that
    # matches how the window looked before fullscreen/resize support.
    width, height = WINDOW_WIDTH, WINDOW_HEIGHT

    with Renderer(width=width, height=height, title=WINDOW_TITLE) as renderer:
        # 3. Wrap the generated grid + game_state for drawing (maze walls,
        # goal, and player marker).
        maze_renderer = MazeRenderer(
            renderer=renderer,
            grid=maze.grid,
            fill_fraction=MAZE_FILL_FRACTION,
            game_state=game_state,
            wall_thickness=WALL_THICKNESS,
            wall_color=WALL_COLOR,
            wall_outline_color=WALL_OUTLINE_COLOR,
            wall_outline_thickness=WALL_OUTLINE_THICKNESS,
            wall_highlight_strength=WALL_HIGHLIGHT_STRENGTH,
            wall_shadow_strength=WALL_SHADOW_STRENGTH,
            wall_lighting_thickness=WALL_LIGHTING_THICKNESS,
            floor_color=FLOOR_COLOR,
        )

        # 4. Input: translates key presses into game_state.move() calls,
        # and separately reports restart requests. Built once and reused
        # across restarts (see design notes above).
        input_handler = InputHandler(renderer, game_state, move_repeat_delay=MOVE_REPEAT_DELAY)

        # 5. Main loop, until the window closes. Runs continuously through
        # a win (movement simply becomes a no-op once won -- see
        # GameState.move()), so pressing 'R' after winning works exactly
        # the same as pressing it mid-game.
        last_time = renderer.get_time()
        while not renderer.should_close():
            current_time = renderer.get_time()
            dt = current_time - last_time
            last_time = current_time

            renderer.begin_frame()

            # (1) Process input -- may call game_state.move_*() internally.
            input_handler.process_input()

            # Fullscreen toggle: main.py only ever calls
            # renderer.toggle_fullscreen() -- everything about HOW that
            # actually happens (glfw.set_window_monitor, remembering/
            # restoring windowed geometry, preserving the GL context)
            # lives entirely in Renderer. InputHandler only reports that
            # F11 was newly pressed; it has no GLFW knowledge itself.
            if input_handler.is_fullscreen_toggle_requested():
                renderer.toggle_fullscreen()

            # Restart: rebuild the maze + game_state locally (no globals),
            # then repoint the existing MazeRenderer/InputHandler at them.
            if input_handler.is_restart_requested():
                maze, game_state = build_session()
                maze_renderer.grid = maze.grid
                maze_renderer.rows = maze.rows
                maze_renderer.cols = maze.cols
                maze_renderer.game_state = game_state
                input_handler.game_state = game_state
                last_time = renderer.get_time()  # avoid a stale dt spike
                print("New maze generated -- good luck!")

            # (2) Update game state -- advance the elapsed-time placeholder.
            game_state.tick(dt)

            # (3) Render the maze, goal, and player.
            maze_renderer.render()

            # Live HUD in the title bar -- no font-rendering library needed.
            status = " -- Solved! Press R to play again" if game_state.won else ""
            renderer.set_title(
                f"{WINDOW_TITLE} | Moves: {game_state.move_count} | "
                f"Time: {int(game_state.elapsed_time)}s{status}"
            )

            # (4) Swap buffers and poll events.
            renderer.end_frame()


if __name__ == "__main__":
    main()
