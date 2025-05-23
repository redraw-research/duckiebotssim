import numpy as np
import pyglet
from pyglet import gl

# Rendering window size

DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600


class ImageRenderer:

    def __init__(self, height: int = DEFAULT_WINDOW_HEIGHT, width: int = DEFAULT_WINDOW_WIDTH):
        self.window = None
        self._height = height
        self._width = width

    def render_cv2_image(self, cv2_image: np.ndarray, channel_order: str = "RGB"):

        if self.window is None:
            context = pyglet.gl.current_context
            self.window = pyglet.window.Window(width=self._width, height=self._height)

        self.window.switch_to()
        self.window.dispatch_events()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self._width, self._height)

        self.window.clear()

        # Setup orthogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, self._width, 0, self._height, 0, 10)

        # Draw the image to the rendering window
        width = cv2_image.shape[1]
        height = cv2_image.shape[0]
        imgData = pyglet.image.ImageData(
            width,
            height,
            channel_order,
            cv2_image[::-1, :, :].tobytes(),
            pitch=width * 3,
        )
        imgData.blit(0, 0, 0, self._width, self._height)

        self.window.flip()
        return imgData

    def close(self):
        if self.window:
            try:
                self.window.close()
            except ImportError:
                pass
