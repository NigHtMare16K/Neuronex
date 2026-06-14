"""
video_source.py
Reads frames from a webcam or video file and pushes them
into the FrameBuffer asynchronously (Thread 1).

Webcam fix:
  - Tries multiple backend APIs (CAP_DSHOW on Windows, CAP_V4L2 on Linux)
  - Retries on transient open failure
  - Falls back gracefully with a clear error message
  - Auto-loops video files when EOF is reached
"""

import cv2
import sys
import time
import threading
import numpy as np

from input.frame_buffer import FrameBuffer
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS

# OpenCV backend priority list — tried in order until one works
_CAM_BACKENDS = [
    cv2.CAP_ANY,       # Let OpenCV decide (usually works)
    cv2.CAP_DSHOW,     # Windows DirectShow
    cv2.CAP_V4L2,      # Linux V4L2
    cv2.CAP_AVFOUNDATION,  # macOS
]

_OPEN_RETRIES   = 3
_RETRY_DELAY_S  = 1.0


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
        self._is_file = isinstance(source, str)
        self._thread  = threading.Thread(target=self._capture_loop,
                                         daemon=True, name="VideoSource")
        self._frame_interval = 1.0 / TARGET_FPS

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._cap = self._open_capture()
        if self._cap is None:
            msg = (
                f"\n[VideoSource] ERROR: Could not open source '{self.source}'.\n"
                f"  • For webcam: ensure it is connected and not used by another app.\n"
                f"  • On Linux: check 'ls /dev/video*' and try index 1 or 2.\n"
                f"  • On Windows: run as administrator or check Device Manager.\n"
                f"  • Alternatively run without --source to use synthetic mode.\n"
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self._running = True
        self._thread.start()
        print(f"[VideoSource] Started → source={self.source}")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        print("[VideoSource] Stopped.")

    # ── Open with retry + backend fallback ────────────────────────
    def _open_capture(self) -> cv2.VideoCapture | None:
        source = self.source

        # For webcam index, try multiple backends
        backends = _CAM_BACKENDS if isinstance(source, int) else [cv2.CAP_ANY]

        for attempt in range(_OPEN_RETRIES):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(source, backend)
                    if cap.isOpened():
                        # Apply settings
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)  # Reduce latency

                        # Verify by reading one frame
                        ret, _ = cap.read()
                        if ret:
                            print(f"[VideoSource] Opened with backend={backend} "
                                  f"on attempt {attempt+1}")
                            return cap
                        cap.release()
                except Exception as e:
                    print(f"[VideoSource] Backend {backend} failed: {e}")

            if attempt < _OPEN_RETRIES - 1:
                print(f"[VideoSource] Retry {attempt+1}/{_OPEN_RETRIES} "
                      f"in {_RETRY_DELAY_S}s...")
                time.sleep(_RETRY_DELAY_S)

        return None

    # ── Capture loop ───────────────────────────────────────────────
    def _capture_loop(self) -> None:
        last_time     = time.perf_counter()
        consecutive_fails = 0

        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                consecutive_fails += 1
                if self._is_file:
                    # EOF — loop back
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_fails = 0
                    continue
                elif consecutive_fails > 30:
                    print("[VideoSource] Too many consecutive read failures. Stopping.",
                          file=sys.stderr)
                    self._running = False
                    break
                time.sleep(0.033)
                continue

            consecutive_fails = 0

            # Resize to target resolution
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
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

    @property
    def is_running(self) -> bool:
        return self._running

