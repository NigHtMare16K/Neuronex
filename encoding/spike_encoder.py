"""
spike_encoder.py
Thread 2 — Converts raw grayscale frames into sparse SpikeEvents.

Algorithm:
  delta(x,y) = current_frame(x,y) - prev_frame(x,y)
  if |delta| > SPIKE_THRESHOLD:
      emit SpikeEvent(x, y, t, polarity=sign(delta))

This is the "artificial retina" layer — biologically analogous to
photoreceptor → bipolar cell signaling in the human eye.
"""

import time
import threading
from typing import Callable
import numpy as np

from input.frame_buffer    import FrameBuffer
from encoding.event_stream import EventQueue, SpikeEvent
from config.settings import (
    SPIKE_THRESHOLD, POLARITY_ON, POLARITY_OFF,
    FRAME_WIDTH, FRAME_HEIGHT
)


class SpikeEncoder:
    """
    Consumes frames from FrameBuffer.
    Produces SpikeEvents into EventQueue.
    Optionally calls on_spikes(events) callback after each frame (for visualiser).
    Runs asynchronously in its own thread.
    """

    def __init__(self,
                 frame_buffer  : FrameBuffer,
                 event_queue   : EventQueue,
                 on_spikes     : Callable[[list[SpikeEvent]], None] | None = None):
        self.frame_buffer  = frame_buffer
        self.event_queue   = event_queue
        self._on_spikes    = on_spikes          # optional callback → Visualizer

        self._prev_frame   = np.zeros((FRAME_HEIGHT, FRAME_WIDTH),
                                       dtype=np.int16)
        self._running      = False
        self._thread       = threading.Thread(target=self._encode_loop,
                                              daemon=True, name="SpikeEncoder")

        # Stats
        self._frames_processed = 0
        self._total_spikes     = 0
        self._total_pixels     = 0

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._thread.start()
        print("[SpikeEncoder] Started.")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        print("[SpikeEncoder] Stopped.")

    # ── Encode loop ────────────────────────────────────────────────
    def _encode_loop(self) -> None:
        while self._running:
            frame = self.frame_buffer.get(timeout=0.1)
            if frame is None:
                continue

            spikes = self._encode_frame(frame)
            self.event_queue.put_batch(spikes)

            # Notify visualiser (non-blocking — copy already done in put_batch)
            if self._on_spikes and spikes:
                try:
                    self._on_spikes(spikes)
                except Exception:
                    pass

            self._frames_processed += 1
            self._total_spikes     += len(spikes)
            self._total_pixels     += frame.size

    def _encode_frame(self, frame: np.ndarray) -> list[SpikeEvent]:
        """
        Core spike encoding — pure NumPy, no ML.
        Returns list of SpikeEvent for pixels whose intensity changed
        more than SPIKE_THRESHOLD since the last frame.
        """
        current = frame.astype(np.int16)
        delta   = current - self._prev_frame           # signed difference

        on_mask  = delta >  SPIKE_THRESHOLD
        off_mask = delta < -SPIKE_THRESHOLD

        t_now = time.perf_counter() * 1000.0

        spikes: list[SpikeEvent] = []

        ys, xs = np.where(on_mask)
        for y, x in zip(ys, xs):
            spikes.append(SpikeEvent(x=int(x), y=int(y),
                                     t=t_now, polarity=POLARITY_ON))

        ys, xs = np.where(off_mask)
        for y, x in zip(ys, xs):
            spikes.append(SpikeEvent(x=int(x), y=int(y),
                                     t=t_now, polarity=POLARITY_OFF))

        self._prev_frame = current
        return spikes

    # ── Stats ──────────────────────────────────────────────────────
    def sparsity(self) -> float:
        if self._total_pixels == 0:
            return 0.0
        return 1.0 - (self._total_spikes / self._total_pixels)

    def stats(self) -> dict:
        return {
            "frames_processed": self._frames_processed,
            "total_spikes"    : self._total_spikes,
            "sparsity"        : f"{self.sparsity() * 100:.1f}%",
        }
