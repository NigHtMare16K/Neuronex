"""
video_source.py
Reads frames from a webcam or video file and pushes them
into the FrameBuffer asynchronously (Thread 1).
"""

import cv2
import time
import threading
import numpy as np

from input.frame_buffer import FrameBuffer
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class VideoSource:
    """
    Captures frames from a camera index (int) or a file path (str).
    Runs in its own daemon thread so it never blocks the main pipeline.
    """

    def __init__(self, source: int | str, buffer: FrameBuffer):
        self.source   = source
        self.buffer   = buffer
        self._cap     = None
        self._running = False
        self._thread  = threading.Thread(target=self._capture_loop,
                                         daemon=True, name="VideoSource")
        self._frame_interval = 1.0 / TARGET_FPS

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        self._running = True
        self._thread.start()
        print(f"[VideoSource] Started → source={self.source}")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("[VideoSource] Stopped.")

    # ── Capture loop ───────────────────────────────────────────────
    def _capture_loop(self) -> None:
        last_time = time.perf_counter()

        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                # End of file — loop back to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize to target resolution
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Convert to grayscale — spike encoder only needs intensity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.buffer.put(gray)

            # Pace to TARGET_FPS
            elapsed = time.perf_counter() - last_time
            sleep_t = self._frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_time = time.perf_counter()

    # ── Info ───────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._running
