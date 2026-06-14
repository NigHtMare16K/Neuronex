"""
frame_buffer.py
Ring buffer that holds the last N frames.
Thread-safe producer/consumer via a lock.
"""

import threading
import numpy as np
from collections import deque
from config.settings import FRAME_BUFFER_SIZE


class FrameBuffer:
    """
    Fixed-size circular buffer for raw video frames.
    Producer (video_source) writes; consumer (spike_encoder) reads.
    """

    def __init__(self, maxsize: int = FRAME_BUFFER_SIZE):
        self._buf  = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    # ── Write ──────────────────────────────────────────────────────
    def put(self, frame: np.ndarray) -> None:
        """Push a new frame. Oldest frame is silently dropped when full."""
        with self._cond:
            self._buf.append(frame.copy())
            self._cond.notify_all()

    # ── Read ───────────────────────────────────────────────────────
    def get(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Block until a frame is available, then pop and return it.
        Returns None on timeout.
        """
        with self._cond:
            if not self._buf:
                self._cond.wait(timeout=timeout)
            if self._buf:
                return self._buf.popleft()
            return None

    def peek_latest(self) -> np.ndarray | None:
        """Return most recent frame without removing it."""
        with self._lock:
            return self._buf[-1].copy() if self._buf else None

    # ── Status ─────────────────────────────────────────────────────
    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buf) == 0
