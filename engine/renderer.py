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
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

    def _init_window(self) -> None:
        """Create the window and make its GL context current."""
        self.window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # enable vsync

    def _init_gl(self) -> None:
        """
        Set up an orthographic 2D projection matching pixel coordinates,
        with (0, 0) at the top-left and y increasing downward.
        """
        glViewport(0, 0, self.width, self.height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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
