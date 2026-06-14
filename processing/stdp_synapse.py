"""
stdp_synapse.py
Spike-Timing-Dependent Plasticity (STDP) weight updater.

Biologically inspired Hebbian learning rule:
  - Pre fires BEFORE post  → strengthen synapse (causal)  → Δw = +A+ * exp(-Δt/τ)
  - Pre fires AFTER  post  → weaken synapse   (anti-causal) → Δw = -A- * exp(-Δt/τ)

No backpropagation. No global loss function.
Updates are purely LOCAL — each synapse only needs to know
when its pre and post neurons last fired.

This allows the network to self-organise feature detectors
from the statistics of incoming spike patterns.
"""

import numpy as np

from config.settings import (
    STDP_A_PLUS, STDP_A_MINUS, STDP_TAU,
    WEIGHT_MIN, WEIGHT_MAX,
    NEURON_GRID_ROWS, NEURON_GRID_COLS
)


class STDPSynapse:
    """
    Manages synaptic weights for the LIF neuron grid.
    Called after each LIF tick to update weights based on
    pre-synaptic event times vs post-synaptic fire times.
    """

    def __init__(self,
                 rows: int = NEURON_GRID_ROWS,
                 cols: int = NEURON_GRID_COLS):
        self.rows = rows
        self.cols = cols

        # Tracks last pre-synaptic spike time per neuron (ms)
        self._t_pre  = np.full((rows, cols), -np.inf, dtype=np.float64)
        # Tracks last post-synaptic fire time per neuron (ms)
        self._t_post = np.full((rows, cols), -np.inf, dtype=np.float64)

        # Synaptic weight matrix — shared reference with LIFNeuronLayer
        self.weights = np.ones((rows, cols), dtype=np.float32) * 0.5

        # Stats
        self._update_count = 0
        self._potentiation = 0   # LTP events
        self._depression   = 0   # LTD events

    # ── Update ─────────────────────────────────────────────────────
    def record_pre_spike(self, ni: int, nj: int, t: float) -> None:
        """Called when a pre-synaptic input arrives at neuron (ni, nj)."""
        self._t_pre[ni, nj] = t

    def record_pre_spikes_batch(self, events: list, pixel_to_neuron_fn) -> None:
        """Batch update pre-spike times from a list of SpikeEvents."""
        for event in events:
            ni, nj = pixel_to_neuron_fn(event.x, event.y)
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                self._t_pre[ni, nj] = event.t

    def apply_stdp(self, fired_map: np.ndarray, t_now: float) -> None:
        """
        Apply STDP weight updates for all neurons that just fired.

        fired_map : bool array (rows, cols) — True where neuron fired this tick
        t_now     : current timestamp in ms
        """
        if not fired_map.any():
            return

        # Record post-synaptic fire times
        self._t_post[fired_map] = t_now

        rows_fired, cols_fired = np.where(fired_map)

        for r, c in zip(rows_fired, cols_fired):
            dt = self._t_pre[r, c] - self._t_post[r, c]   # t_pre - t_post

            if np.isfinite(dt):
                if dt < 0:
                    # Pre fired BEFORE post → Long-Term Potentiation (LTP)
                    dw = STDP_A_PLUS * np.exp(dt / STDP_TAU)
                    self.weights[r, c] += dw
                    self._potentiation += 1
                else:
                    # Pre fired AFTER post → Long-Term Depression (LTD)
                    dw = STDP_A_MINUS * np.exp(-dt / STDP_TAU)
                    self.weights[r, c] -= dw
                    self._depression += 1

                self._update_count += 1

        # Clip weights to biological bounds
        np.clip(self.weights, WEIGHT_MIN, WEIGHT_MAX, out=self.weights)

    # ── Stats / Introspection ──────────────────────────────────────
    def stats(self) -> dict:
        return {
            "total_updates" : self._update_count,
            "potentiation"  : self._potentiation,
            "depression"    : self._depression,
            "weight_mean"   : float(self.weights.mean()),
            "weight_std"    : float(self.weights.std()),
        }

    def weight_map(self) -> np.ndarray:
        """Return copy of current weight matrix for visualisation."""
        return self.weights.copy()

    def reset_weights(self) -> None:
        self.weights[:] = 0.5
        self._t_pre[:]  = -np.inf
        self._t_post[:] = -np.inf
