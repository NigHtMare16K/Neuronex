"""
lif_neuron.py
Leaky Integrate-and-Fire (LIF) neuron grid — Thread 3.

Each neuron (i, j) in the grid:
  1. Accumulates membrane potential from incoming spikes
  2. Leaks toward rest at each tick
  3. Fires an output spike when membrane crosses V_THRESH
  4. Enters refractory period after firing

Equation (discrete):
  V[t] = V[t-1] * LEAK_FACTOR + weighted_input
  if V[t] >= V_THRESH: fire(); V[t] = V_RESET
"""

import time
import threading
import numpy as np

from encoding.event_stream   import EventQueue, SpikeEvent
from processing.time_surface import TimeSurface
from config.settings import (
    NEURON_GRID_ROWS, NEURON_GRID_COLS,
    LEAK_FACTOR, V_THRESH, V_RESET, V_REST,
    REFRACTORY_MS, FRAME_WIDTH, FRAME_HEIGHT
)


class LIFNeuronLayer:
    """
    2D grid of LIF neurons mapped over the frame.
    Accepts an optional STDPSynapse instance to share weights.
    """

    def __init__(self,
                 event_queue  : EventQueue,
                 time_surface : TimeSurface,
                 stdp         = None,          # STDPSynapse | None
                 rows: int    = NEURON_GRID_ROWS,
                 cols: int    = NEURON_GRID_COLS):

        self.event_queue  = event_queue
        self.time_surface = time_surface
        self.stdp         = stdp
        self.rows = rows
        self.cols = cols

        self.patch_h = FRAME_HEIGHT // rows
        self.patch_w = FRAME_WIDTH  // cols

        # State arrays
        self.membrane   = np.full((rows, cols), V_REST, dtype=np.float32)
        self.refractory = np.zeros((rows, cols), dtype=np.float32)
        self.fired      = np.zeros((rows, cols), dtype=bool)

        # Use shared weights from STDP if available, else own weights
        if stdp is not None:
            self.weights = stdp.weights      # shared reference
        else:
            self.weights = np.ones((rows, cols), dtype=np.float32) * 0.5

        # Output spike log
        self._spike_log : list[tuple[int, int, float]] = []
        self._log_lock  = threading.Lock()
        self._total_fires = 0

        self._running      = False
        self._tick_interval = 0.005   # 5 ms tick
        self._thread = threading.Thread(target=self._process_loop,
                                        daemon=True, name="LIFNeuronLayer")

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._thread.start()
        print(f"[LIFNeuronLayer] Started — grid {self.rows}×{self.cols}")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        print("[LIFNeuronLayer] Stopped.")

    # ── Processing loop ────────────────────────────────────────────
    def _process_loop(self) -> None:
        last_tick = time.perf_counter()

        while self._running:
            events = self.event_queue.get_batch(max_events=200)
            t_now  = time.perf_counter() * 1000.0

            for event in events:
                self.time_surface.update(event)
                ni, nj = self._pixel_to_neuron(event.x, event.y)
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    if self.refractory[ni, nj] <= 0:
                        context = self.time_surface.local_context(
                            event.x, event.y, polarity=event.polarity)
                        self.membrane[ni, nj] += (
                            self.weights[ni, nj] * (1.0 + context)
                        )

            self._tick(t_now)

            # Notify STDP of pre-spike times and apply weight update
            if self.stdp is not None and events:
                self.stdp.record_pre_spikes_batch(events, self._pixel_to_neuron)
                if self.fired.any():
                    self.stdp.apply_stdp(self.fired, t_now)
                    # Sync weights back
                    self.weights[:] = self.stdp.weights

            elapsed = time.perf_counter() - last_tick
            sleep_t = self._tick_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_tick = time.perf_counter()

    def _tick(self, t_now: float) -> None:
        tick_ms = self._tick_interval * 1000.0
        self.refractory = np.maximum(0.0, self.refractory - tick_ms)
        active_mask     = self.refractory <= 0

        self.membrane = np.where(active_mask,
                                 self.membrane * LEAK_FACTOR,
                                 V_RESET)

        self.fired = (self.membrane >= V_THRESH) & active_mask

        if self.fired.any():
            rows_fired, cols_fired = np.where(self.fired)
            with self._log_lock:
                for r, c in zip(rows_fired, cols_fired):
                    self._spike_log.append((int(r), int(c), t_now))
            self._total_fires       += int(self.fired.sum())
            self.membrane[self.fired]   = V_RESET
            self.refractory[self.fired] = float(REFRACTORY_MS)

    # ── Helpers ────────────────────────────────────────────────────
    def _pixel_to_neuron(self, x: int, y: int) -> tuple[int, int]:
        ni = min(y // self.patch_h, self.rows - 1)
        nj = min(x // self.patch_w, self.cols - 1)
        return ni, nj

    # ── Public API ─────────────────────────────────────────────────
    def get_fired_map(self) -> np.ndarray:
        return self.fired.copy()

    def get_membrane_map(self) -> np.ndarray:
        return self.membrane.copy()

    def drain_spike_log(self) -> list[tuple[int, int, float]]:
        with self._log_lock:
            log = self._spike_log.copy()
            self._spike_log.clear()
        return log

    def total_output_spikes(self) -> int:
        return self._total_fires

