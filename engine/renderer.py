"""
engine/renderer.py

Generic 2D rendering layer built on GLFW (window/context/input) and
PyOpenGL (drawing). This is the ONLY file that is allowed to know about
GLFW or OpenGL calls in this project -- everything else should go through
the Renderer class.

Design notes (why it's built this way):
- Zero maze knowledge. This module has no concept of cells, walls, grids,
  or rows/cols. It only knows how to open a window, set up a 2D coordinate
  system, clear the screen, and draw primitive shapes (lines, rectangles).
  A future maze-rendering module will import Renderer and call
  draw_line()/draw_rect() in a loop over the maze grid -- that loop and any
  "how do I draw a cell" logic belongs there, never here. This keeps
  rendering and maze-generation permanently decoupled, in both directions:
  generator.py/cell.py have zero imports from this file, and this file has
  zero imports from maze.*.
- Top-left origin, y-down. glOrtho is set up as (0, width, height, 0, ...)
  rather than the math convention (0, width, 0, height, ...), so (0, 0) is
  the top-left corner and y increases downward. This matches how grids are
  normally indexed (row 0 at the top, row increasing downward) and how
  print_ascii() in generator.py already prints -- so screen coordinates and
  grid coordinates will agree once the maze layer is wired in, with no sign
  flips needed.
- Fixed-function pipeline (glBegin/glEnd, glOrtho) rather than shaders/VBOs.
  This is a 2D line-and-rectangle renderer for a maze; a compatibility
  context keeps the code simple and readable. If a future version needs
  shader-based rendering, that's a change localized entirely to this file --
  callers only use draw_line()/draw_rect()/begin_frame()/end_frame(), so
  the public interface would not need to change.
- One instance = one window. All state (window handle, dimensions, clear
  color) lives on the Renderer instance -- no globals or module-level
  window handles -- so nothing here fights over shared mutable state.
- Context-manager support (__enter__/__exit__) is provided so callers can
  use `with Renderer(...) as r:` and always clean up the GLFW window/context
  even if an exception happens mid-loop.
- Fullscreen uses GLFW's proper API, not window recreation.
  toggle_fullscreen() calls glfw.set_window_monitor() on the SAME window
  handle -- it never destroys/recreates the window or the GL context, so
  no textures, GL state, or the window's identity are ever lost across the
  switch. Windowed-mode position/size are remembered on the instance
  (self._windowed_pos/_windowed_size) so returning from fullscreen restores
  exactly where the window was, rather than resetting to some default.
- Resize is handled by ONE mechanism, used for every size change. Whether
  the size change comes from toggle_fullscreen(), the user dragging an
  edge, or the OS itself, the same _apply_viewport_and_projection() runs
  and self.width/self.height get resynced from the real framebuffer size
  (not the requested size) -- so callers (MazeRenderer) always see
  Renderer's current, correct dimensions, however the window got there.
"""

from typing import Optional, Tuple

import glfw
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_LINE_LOOP,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_QUADS,
    GL_SRC_ALPHA,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glEnable,
    glEnd,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glVertex2f,
    glViewport,
)

# RGB color, each channel in [0.0, 1.0].
Color = Tuple[float, float, float]

# String key names -> GLFW key constants. Kept private to this module so
# that callers (e.g. a future input_handler.py) can ask Renderer
# "is 'UP' pressed?" using plain strings, without ever importing glfw or
# knowing its constants themselves -- GLFW stays encapsulated entirely
# inside this file, per the module docstring above.
_KEY_NAME_MAP = {
    "UP": glfw.KEY_UP,
    "DOWN": glfw.KEY_DOWN,
    "LEFT": glfw.KEY_LEFT,
    "RIGHT": glfw.KEY_RIGHT,
    "W": glfw.KEY_W,
    "A": glfw.KEY_A,
    "S": glfw.KEY_S,
    "D": glfw.KEY_D,
    "R": glfw.KEY_R,
    "N": glfw.KEY_N,
    "P": glfw.KEY_P,
    "F11": glfw.KEY_F11,
    "ESCAPE": glfw.KEY_ESCAPE,
    "SPACE": glfw.KEY_SPACE,
    "ENTER": glfw.KEY_ENTER,
}


class Renderer:
    """
    Owns a GLFW window + OpenGL context and exposes generic 2D drawing
    primitives (lines, rectangles). Has no knowledge of what is being drawn.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        title: str = "Renderer",
        background_color: Color = (0.07, 0.07, 0.09),
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.title: str = title
        self.background_color: Color = background_color

        # GLFW window handle. Typed loosely (Optional[object]) since glfw's
        # window handle is an opaque C pointer wrapper, not a documented
        # Python type.
        self.window: Optional[object] = None

        # Fullscreen state. _windowed_pos/_windowed_size remember the
        # window's windowed-mode geometry so toggle_fullscreen() can
        # restore it exactly when returning from fullscreen -- captured
        # fresh each time we ENTER fullscreen (see toggle_fullscreen()),
        # and initialized here to the window's starting position/size so
        # there's always something sane to restore even if F11 is the
        # very first thing pressed.
        self._is_fullscreen: bool = False
        self._windowed_pos: Tuple[int, int] = (0, 0)
        self._windowed_size: Tuple[int, int] = (width, height)

        self._init_glfw()
        self._init_window()
        self._init_gl()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_glfw(self) -> None:
        """Initialize the GLFW library itself (once per process)."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # We use the legacy fixed-function pipeline (glBegin/glEnd), so we
        # deliberately do NOT request a core profile -- a core context would
        # reject those calls.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        # Resizable, so both user drag-resize AND the fullscreen toggle
        # (which resizes the same window rather than replacing it) work.
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    def _init_window(self) -> None:
        """Create the window, make its GL context current, and remember
        its starting geometry + register the resize callback."""
        self.window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # enable vsync

        # Renderer's width/height must reflect the real FRAMEBUFFER size
        # (pixels), not the requested window size (screen coordinates) --
        # these can differ on HiDPI/Retina displays. Everything this class
        # draws is in pixel coordinates, so framebuffer size is what
        # matters for glViewport/glOrtho and for anything (MazeRenderer)
        # that reads self.width/self.height.
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.width, self.height = fb_width, fb_height
        self._windowed_size = (fb_width, fb_height)
        self._windowed_pos = glfw.get_window_pos(self.window)

        # ONE callback handles every resize, regardless of cause (user
        # drag, OS action, or toggle_fullscreen()'s own set_window_monitor
        # call) -- so there is a single, always-correct path that keeps
        # self.width/self.height and the GL viewport/projection in sync,
        # rather than several call sites each trying to update them.
        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_resize)

    def _on_framebuffer_resize(self, window: object, width: int, height: int) -> None:
        """GLFW callback: fired whenever the framebuffer size changes, for
        any reason. Resyncs Renderer's dimensions and the GL viewport/
        projection so they're never stale relative to the real window."""
        self.width, self.height = width, height
        self._apply_viewport_and_projection()

    def _apply_viewport_and_projection(self) -> None:
        """
        Set up an orthographic 2D projection matching pixel coordinates,
        with (0, 0) at the top-left and y increasing downward, sized to
        self.width/self.height. Called once at startup (_init_gl) and
        again every time the framebuffer size changes (see
        _on_framebuffer_resize) -- pulled out into its own method
        specifically so both call sites share one implementation instead
        of the projection math being duplicated.
        """
        glViewport(0, 0, self.width, self.height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _init_gl(self) -> None:
        """One-time GL setup: the initial viewport/projection, plus
        alpha blending (not resize-dependent, so it lives here rather
        than in _apply_viewport_and_projection, which reruns on every
        resize)."""
        self._apply_viewport_and_projection()

        # Alpha blending, so draw_line/draw_rect calls can use translucent
        # colors later (e.g. highlighting a solver path) without every
        # caller having to remember to enable this themselves.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def should_close(self) -> bool:
        """True once the user has requested the window close (e.g. clicked X)."""
        return bool(glfw.window_should_close(self.window))

    def is_key_pressed(self, key_name: str) -> bool:
        """
        Return True if the named key is currently held down. Key names are
        plain strings ('UP', 'DOWN', 'LEFT', 'RIGHT', 'W', 'A', 'S', 'D',
        'ESCAPE', 'SPACE', 'ENTER') -- callers never need to import glfw or
        know its key constants; this method is the one place that
        translates a human-readable name into a GLFW key code.
        """
        key = _KEY_NAME_MAP.get(key_name.upper())
        if key is None:
            raise ValueError(f"Unknown key name: {key_name!r}")
        return glfw.get_key(self.window, key) == glfw.PRESS

    def get_time(self) -> float:
        """
        Return the number of seconds since GLFW was initialized (a
        monotonic clock). Exposed so callers (e.g. main.py's loop) can
        compute a frame's dt for time-based updates -- like
        GameState.tick(dt) -- without importing glfw themselves.
        """
        return glfw.get_time()

    def set_title(self, title: str) -> None:
        """
        Change the window's title bar text. Exposed specifically so
        callers can show live info (move count, elapsed time, etc.) in the
        title bar without this project needing a text-rendering/font
        library yet -- the OS-drawn title bar is "free" text rendering.
        """
        glfw.set_window_title(self.window, title)

    @property
    def is_fullscreen(self) -> bool:
        """True if the window is currently fullscreen."""
        return self._is_fullscreen

    def toggle_fullscreen(self) -> None:
        """
        Toggle between windowed and fullscreen mode on the SAME window,
        using glfw.set_window_monitor() -- never destroying/recreating the
        window, so the GL context (and everything tied to it) is preserved
        across the switch, exactly as required.

        Entering fullscreen: remembers the current windowed position/size
        (so it can be restored later), then hands the window to the
        primary monitor at that monitor's current video mode -- native
        resolution and refresh rate, not a hardcoded size.

        Returning to windowed: hands the window back with monitor=None,
        using the exact position/size that were remembered when fullscreen
        was entered -- so the window reappears exactly where it was, not
        at some default location.

        Either way, self.width/self.height and the GL viewport/projection
        get updated via the SAME path as any other resize
        (_on_framebuffer_resize / _apply_viewport_and_projection) -- this
        method doesn't duplicate that logic, it just triggers it and
        (defensively) re-applies it immediately after, since GLFW does not
        guarantee the resize callback fires synchronously before the next
        poll_events().
        """
        if self._is_fullscreen:
            x, y = self._windowed_pos
            w, h = self._windowed_size
            glfw.set_window_monitor(self.window, None, x, y, w, h, glfw.DONT_CARE)
            self._is_fullscreen = False
        else:
            # Remember exactly where the window was, so windowed mode can
            # be restored precisely later.
            self._windowed_pos = glfw.get_window_pos(self.window)
            self._windowed_size = glfw.get_window_size(self.window)

            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window, monitor, 0, 0,
                mode.size.width, mode.size.height, mode.refresh_rate,
            )
            self._is_fullscreen = True

        # Defensive resync -- see docstring above.
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.width, self.height = fb_width, fb_height
        self._apply_viewport_and_projection()

    def begin_frame(self) -> None:
        """Clear the screen with the configured background color."""
        r, g, b = self.background_color
        glClearColor(r, g, b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_frame(self) -> None:
        """Present the frame and process window/input events."""
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        """Destroy the window and terminate GLFW. Safe to call more than once."""
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def draw_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: Color = (1.0, 1.0, 1.0),
        line_width: float = 1.0,
    ) -> None:
        """Draw a single line segment from (x1, y1) to (x2, y2) in pixel coords."""
        glColor3f(*color)
        glLineWidth(line_width)
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()

    def draw_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        color: Color = (1.0, 1.0, 1.0),
        filled: bool = True,
    ) -> None:
        """
        Draw a rectangle with its top-left corner at (x, y) and size (w, h)
        in pixel coordinates. Filled by default; pass filled=False for an
        unfilled outline (e.g. useful later for a player marker or cell
        highlight border).
        """
        glColor3f(*color)
        mode = GL_QUADS if filled else GL_LINE_LOOP
        glBegin(mode)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()


if __name__ == "__main__":
    # Standalone smoke test: confirms GLFW + PyOpenGL + this module work
    # together, with NO maze logic involved -- just a border rectangle and
    # a crosshair drawn with draw_rect()/draw_line(). If this window shows
    # a bordered box with a plus sign in the middle, the rendering layer is
    # solid and ready for the maze-drawing module to be built on top of it.
    with Renderer(width=800, height=800, title="Renderer smoke test") as renderer:
        while not renderer.should_close():
            renderer.begin_frame()

            # Border rectangle (unfilled), 40px in from each edge.
            renderer.draw_rect(
                40, 40, renderer.width - 80, renderer.height - 80,
                color=(0.9, 0.9, 0.9), filled=False,
            )

            # Crosshair through the center.
            cx, cy = renderer.width / 2, renderer.height / 2
            renderer.draw_line(cx, 40, cx, renderer.height - 40, color=(0.2, 0.8, 0.4))
            renderer.draw_line(40, cy, renderer.width - 40, cy, color=(0.2, 0.8, 0.4))

            renderer.end_frame()
