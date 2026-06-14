"""
synthetic_source.py
Generates motion patterns in-memory when no webcam or video file is available.
Ideal for hackathon demos and CI/benchmark runs.
"""

import time
import threading
import numpy as np

from input.frame_buffer import FrameBuffer
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class SyntheticVideoSource:
    """
    Renders moving blobs on a dark background — produces rich spike activity
    without requiring external hardware.
    """

    def __init__(self, buffer: FrameBuffer):
        self.buffer   = buffer
        self._running = False
        self._thread  = threading.Thread(target=self._render_loop,
                                         daemon=True, name="SyntheticVideoSource")
        self._frame_interval = 1.0 / TARGET_FPS
        self._t = 0.0

    def start(self) -> None:
        self._running = True
        self._thread.start()
        print("[SyntheticVideoSource] Started — procedural motion demo")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        print("[SyntheticVideoSource] Stopped.")

    def _render_loop(self) -> None:
        last_time = time.perf_counter()

        while self._running:
            frame = self._render_frame(self._t)
            self.buffer.put(frame)
            self._t += 1.0 / TARGET_FPS

            elapsed = time.perf_counter() - last_time
            sleep_t = self._frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_time = time.perf_counter()

    def _render_frame(self, t: float) -> np.ndarray:
        """Two moving circles + a sweeping edge — strong temporal contrast."""
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 30, dtype=np.uint8)

        # Blob 1 — horizontal sweep
        x1 = int((FRAME_WIDTH  * 0.5) + (FRAME_WIDTH  * 0.35) * np.sin(t * 1.2))
        y1 = int(FRAME_HEIGHT * 0.4)
        cv2_circle(frame, x1, y1, 28, 220)

        # Blob 2 — vertical bounce
        x2 = int(FRAME_WIDTH * 0.75)
        y2 = int((FRAME_HEIGHT * 0.5) + (FRAME_HEIGHT * 0.3) * np.sin(t * 2.0))
        cv2_circle(frame, x2, y2, 22, 180)

        # Moving vertical bar (edge-like motion)
        bar_x = int((t * 60) % (FRAME_WIDTH + 40)) - 20
        frame[:, max(0, bar_x):min(FRAME_WIDTH, bar_x + 8)] = 200

        return frame


def cv2_circle(img: np.ndarray, cx: int, cy: int, r: int, val: int) -> None:
    """Lightweight filled circle without OpenCV dependency."""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = val
