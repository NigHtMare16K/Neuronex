"""
lif_neuron.py
Leaky Integrate-and-Fire (LIF) neuron grid — Thread 3.

Each neuron (i, j) in the grid:
  1. Accumulates membrane potential from incoming spikes
  2. Leaks toward rest at each tick
  3. Fires an output spike when membrane crosses V_THRESH
  4. Enters refractory period after firing

Equation:
  dV/dt = -V/τ_m + I_spike(t)

Discrete form (per tick):
  V[t] = V[t-1] * LEAK_FACTOR + weighted_input
  if V[t] >= V_THRESH:
      fire(); V[t] = V_RESET; refractory_timer = REFRACTORY_MS
"""

import time
import threading
import numpy as np

from encoding.event_stream    import EventQueue, SpikeEvent
from processing.time_surface  import TimeSurface
from processing.stdp_synapse  import STDPSynapse
from config.settings import (
    NEURON_GRID_ROWS, NEURON_GRID_COLS,
    LEAK_FACTOR, V_THRESH, V_RESET, V_REST,
    REFRACTORY_MS, FRAME_WIDTH, FRAME_HEIGHT
)


class LIFNeuronLayer:
    """
    2D grid of LIF neurons mapped over the frame.
    Each neuron covers a receptive field of (patch_h × patch_w) pixels.
    Runs asynchronously in its own thread.
    """

    def __init__(self,
                 event_queue:  EventQueue,
                 time_surface: TimeSurface,
                 stdp:         STDPSynapse | None = None,
                 rows: int = NEURON_GRID_ROWS,
                 cols: int = NEURON_GRID_COLS):

        self.event_queue  = event_queue
        self.time_surface = time_surface
        self.stdp         = stdp
        self.rows = rows
        self.cols = cols

        # Receptive field size per neuron (pixels)
        self.patch_h = FRAME_HEIGHT // rows
        self.patch_w = FRAME_WIDTH  // cols

        # State arrays — shape (rows, cols)
        self.membrane      = np.full((rows, cols), V_REST, dtype=np.float32)
        self.refractory    = np.zeros((rows, cols), dtype=np.float32)  # ms remaining
        self.fired         = np.zeros((rows, cols), dtype=bool)        # this tick

        # Synaptic weights per neuron (shared with STDP when provided)
        if stdp is not None:
            self.weights = stdp.weights
        else:
            self.weights = np.ones((rows, cols), dtype=np.float32) * 0.5

        # Output spike log: list of (neuron_row, neuron_col, timestamp)
        self._spike_log    : list[tuple[int, int, float]] = []
        self._log_lock     = threading.Lock()

        self._running  = False
        self._thread   = threading.Thread(target=self._process_loop,
                                          daemon=True, name="LIFNeuronLayer")
        self._tick_interval = 0.005   # 5 ms tick (200 Hz processing rate)

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
            # 1. Drain incoming spikes
            events = self.event_queue.get_batch(max_events=200)
            t_now  = time.perf_counter() * 1000.0

            # 2. Accumulate into membrane (one surface refresh per batch)
            if self.stdp and events:
                self.stdp.record_pre_spikes_batch(events, self._pixel_to_neuron)

            on_surf = off_surf = None
            if events:
                on_surf  = self.time_surface.get_surface(1, t_now)
                off_surf = self.time_surface.get_surface(-1, t_now)

            for event in events:
                self.time_surface.update(event)
                ni, nj = self._pixel_to_neuron(event.x, event.y)
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    if self.refractory[ni, nj] <= 0:
                        surf = on_surf if event.is_on() else off_surf
                        context = self._local_mean(surf, event.x, event.y)
                        self.membrane[ni, nj] += (
                            self.weights[ni, nj] * (1.0 + context)
                        )

            # 3. Tick: leak + threshold check
            self._tick(t_now)

            if self.stdp:
                self.stdp.apply_stdp(self.fired, t_now)

            # 4. Pace the loop
            elapsed = time.perf_counter() - last_tick
            sleep_t = self._tick_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_tick = time.perf_counter()

    def _tick(self, t_now: float) -> None:
        """Apply leak, check thresholds, handle refractory periods."""
        tick_ms = self._tick_interval * 1000.0

        # Decrement refractory counters
        self.refractory = np.maximum(0.0, self.refractory - tick_ms)

        # Apply leak only to non-refractory neurons
        active_mask = self.refractory <= 0
        self.membrane = np.where(active_mask,
                                 self.membrane * LEAK_FACTOR,
                                 V_RESET)

        # Fire neurons above threshold
        self.fired = (self.membrane >= V_THRESH) & active_mask

        if self.fired.any():
            rows_fired, cols_fired = np.where(self.fired)
            with self._log_lock:
                for r, c in zip(rows_fired, cols_fired):
                    self._spike_log.append((int(r), int(c), t_now))

            # Reset fired neurons
            self.membrane[self.fired]   = V_RESET
            self.refractory[self.fired] = float(REFRACTORY_MS)

    # ── Helpers ────────────────────────────────────────────────────
    def _local_mean(self, surface: np.ndarray, x: int, y: int,
                    radius: int = 5) -> float:
        h, w = surface.shape
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        patch = surface[y0:y1, x0:x1]
        return float(patch.mean()) if patch.size else 0.0

    def _pixel_to_neuron(self, x: int, y: int) -> tuple[int, int]:
        """Map a pixel coordinate to its LIF neuron index."""
        ni = min(y // self.patch_h, self.rows - 1)
        nj = min(x // self.patch_w, self.cols - 1)
        return ni, nj

    # ── Public API ─────────────────────────────────────────────────
    def get_fired_map(self) -> np.ndarray:
        """Return boolean grid of neurons that fired in the last tick."""
        return self.fired.copy()

    def get_membrane_map(self) -> np.ndarray:
        """Return current membrane potential grid (for visualisation)."""
        return self.membrane.copy()

    def drain_spike_log(self) -> list[tuple[int, int, float]]:
        """Return and clear the accumulated spike log."""
        with self._log_lock:
            log = self._spike_log.copy()
            self._spike_log.clear()
        return log

    def total_output_spikes(self) -> int:
        with self._log_lock:
            return len(self._spike_log)
