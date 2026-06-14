"""
event_stream.py
Defines the SpikeEvent dataclass and the async EventQueue
that connects the encoder to the processing stages.
"""

import queue
import time
from dataclasses import dataclass, field

from config.settings import EVENT_QUEUE_MAXSIZE, POLARITY_ON, POLARITY_OFF


@dataclass(slots=True)
class SpikeEvent:
    """
    A single neuromorphic spike event.

    x, y      : pixel coordinates of the event
    t         : timestamp in milliseconds (float, monotonic)
    polarity  : +1 (ON — brightness increased)
                -1 (OFF — brightness decreased)
    """
    x        : int
    y        : int
    t        : float
    polarity : int

    def is_on(self)  -> bool: return self.polarity == POLARITY_ON
    def is_off(self) -> bool: return self.polarity == POLARITY_OFF

    def __repr__(self) -> str:
        pol = "ON " if self.is_on() else "OFF"
        return f"SpikeEvent({self.x:4d},{self.y:4d} t={self.t:.1f}ms {pol})"


class EventQueue:
    """
    Thread-safe async queue of SpikeEvents.
    Producer: SpikeEncoder  →  Consumer: LIF neuron layer.
    Drops oldest events when full (non-blocking put).
    """

    def __init__(self, maxsize: int = EVENT_QUEUE_MAXSIZE):
        self._q = queue.Queue(maxsize=maxsize)
        self._total_in  = 0
        self._total_out = 0
        self._dropped   = 0

    def put(self, event: SpikeEvent) -> None:
        try:
            self._q.put_nowait(event)
            self._total_in += 1
        except queue.Full:
            self._dropped += 1         # Overload — drop oldest implicitly

    def put_batch(self, events: list[SpikeEvent]) -> None:
        for e in events:
            self.put(e)

    def get(self, timeout: float = 0.05) -> SpikeEvent | None:
        try:
            event = self._q.get(timeout=timeout)
            self._total_out += 1
            return event
        except queue.Empty:
            return None

    def get_batch(self, max_events: int = 100) -> list[SpikeEvent]:
        """Drain up to max_events from the queue non-blocking."""
        batch = []
        for _ in range(max_events):
            try:
                batch.append(self._q.get_nowait())
                self._total_out += 1
            except queue.Empty:
                break
        return batch

    # ── Stats ──────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "queued"  : self._q.qsize(),
            "total_in": self._total_in,
            "total_out": self._total_out,
            "dropped" : self._dropped,
        }

    def qsize(self) -> int:
        return self._q.qsize()
