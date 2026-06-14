"""
time_surface.py
Maintains a 2D spatial memory of the most recent spike time
at each pixel location.

The time surface S(x, y, t) = exp(-(t - t_last(x,y)) / τ)

This gives a continuous "recency" map:
  1.0  → spiked right now
  ~0.0 → spiked long ago (or never)

Used downstream by the LIF layer to weight incoming events
by how recent their spatial neighbourhood was active.
"""

import time
import numpy as np

from config.settings import (
    FRAME_WIDTH, FRAME_HEIGHT, TIME_SURFACE_TAU
)
from encoding.event_stream import SpikeEvent


class TimeSurface:
    """
    Separate ON and OFF time surfaces for full polarity tracking.
    Shape: (FRAME_HEIGHT, FRAME_WIDTH) each.
    """

    def __init__(self,
                 height: int = FRAME_HEIGHT,
                 width:  int = FRAME_WIDTH,
                 tau:    float = TIME_SURFACE_TAU):
        self.height = height
        self.width  = width
        self.tau    = tau

        # Last spike timestamps per pixel (milliseconds), -inf = never spiked
        self._t_on  = np.full((height, width), -np.inf, dtype=np.float64)
        self._t_off = np.full((height, width), -np.inf, dtype=np.float64)

    # ── Update ─────────────────────────────────────────────────────
    def update(self, event: SpikeEvent) -> None:
        """Record the spike time at (x, y) for the correct polarity surface."""
        x, y = event.x, event.y
        if 0 <= x < self.width and 0 <= y < self.height:
            if event.is_on():
                self._t_on[y, x] = event.t
            else:
                self._t_off[y, x] = event.t

    def update_batch(self, events: list[SpikeEvent]) -> None:
        for e in events:
            self.update(e)

    # ── Query ──────────────────────────────────────────────────────
    def get_surface(self, polarity: int, t_now: float | None = None) -> np.ndarray:
        """
        Return the decayed time surface at the current moment.
        Values in [0, 1] where 1 = just spiked.
        """
        if t_now is None:
            t_now = time.perf_counter() * 1000.0

        t_last = self._t_on if polarity > 0 else self._t_off
        delta  = t_now - t_last                         # age of each spike (ms)

        # Exponential decay; pixels that never spiked stay at 0
        surface = np.where(
            np.isfinite(t_last),
            np.exp(-delta / self.tau),
            0.0
        )
        return surface.astype(np.float32)

    def get_combined_surface(self, t_now: float | None = None) -> np.ndarray:
        """ON surface - OFF surface — signed activity map."""
        on  = self.get_surface( 1, t_now)
        off = self.get_surface(-1, t_now)
        return on - off

    def local_context(self, x: int, y: int,
                      radius: int = 5,
                      polarity: int = 1,
                      t_now: float | None = None) -> float:
        """
        Average time-surface value in a small neighbourhood around (x,y).
        Used by LIF neurons to weight their membrane update.
        """
        surface = self.get_surface(polarity, t_now)
        y0 = max(0, y - radius);  y1 = min(self.height, y + radius + 1)
        x0 = max(0, x - radius);  x1 = min(self.width,  x + radius + 1)
        patch = surface[y0:y1, x0:x1]
        return float(patch.mean()) if patch.size > 0 else 0.0

    # ── Reset ──────────────────────────────────────────────────────
    def reset(self) -> None:
        self._t_on[:]  = -np.inf
        self._t_off[:] = -np.inf
