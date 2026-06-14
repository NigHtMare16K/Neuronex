"""
pattern_detector.py
Rule-based pattern detection from LIF output spikes.

No ML. Detection is purely based on:
  1. Spike RATE  — how many neurons fired in the last window?
  2. Spike DENSITY clusters — where are spikes concentrated spatially?
  3. Spike FLOW  — are spike clusters moving directionally?

This maps neuromorphic output to human-readable events:
  MOTION_DETECTED, EDGE_ACTIVITY, STILLNESS, etc.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field

from config.settings import (
    SPIKE_RATE_WINDOW_MS, MOTION_SPIKE_THRESHOLD,
    CLUSTER_MIN_SPIKES, CLUSTER_RADIUS_PX,
    NEURON_GRID_ROWS, NEURON_GRID_COLS,
    FRAME_WIDTH, FRAME_HEIGHT
)


# ── Event types ────────────────────────────────────────────────────
STILLNESS        = "STILLNESS"
MOTION_DETECTED  = "MOTION_DETECTED"
EDGE_ACTIVITY    = "EDGE_ACTIVITY"
HIGH_ACTIVITY    = "HIGH_ACTIVITY"


@dataclass
class MotionRegion:
    """A detected cluster of neuronal activity."""
    center_x   : float        # frame pixel x
    center_y   : float        # frame pixel y
    spike_count: int
    label      : str = MOTION_DETECTED

    def to_bbox(self, radius: int = CLUSTER_RADIUS_PX):
        x0 = max(0, int(self.center_x) - radius)
        y0 = max(0, int(self.center_y) - radius)
        x1 = min(FRAME_WIDTH,  int(self.center_x) + radius)
        y1 = min(FRAME_HEIGHT, int(self.center_y) + radius)
        return (x0, y0, x1, y1)


class PatternDetector:
    """
    Analyses the spike log from LIFNeuronLayer and emits
    structured detection events.
    """

    def __init__(self,
                 rows: int = NEURON_GRID_ROWS,
                 cols: int = NEURON_GRID_COLS):
        self.rows = rows
        self.cols = cols
        self.patch_h = FRAME_HEIGHT // rows
        self.patch_w = FRAME_WIDTH  // cols

        # Rolling spike timestamp buffer for rate calculation
        self._spike_times: deque[float] = deque()

        # Latest detection result
        self.current_state   : str               = STILLNESS
        self.active_regions  : list[MotionRegion] = []

        # History
        self._detection_log  : list[dict] = []

    # ── Main update ────────────────────────────────────────────────
    def update(self, spike_log: list[tuple[int, int, float]]) -> str:
        """
        Feed the latest spike log from LIFNeuronLayer.
        Returns current detection state string.

        spike_log: list of (neuron_row, neuron_col, timestamp_ms)
        """
        t_now = time.perf_counter() * 1000.0

        # 1. Add new spikes to rolling window
        for (r, c, t) in spike_log:
            self._spike_times.append((r, c, t))

        # 2. Prune old spikes outside the time window
        cutoff = t_now - SPIKE_RATE_WINDOW_MS
        while self._spike_times and self._spike_times[0][2] < cutoff:
            self._spike_times.popleft()

        recent = list(self._spike_times)
        spike_count = len(recent)

        # 3. Determine state
        if spike_count < MOTION_SPIKE_THRESHOLD:
            self.current_state  = STILLNESS
            self.active_regions = []
        elif spike_count > MOTION_SPIKE_THRESHOLD * 5:
            self.current_state  = HIGH_ACTIVITY
            self.active_regions = self._cluster(recent)
        else:
            self.current_state  = MOTION_DETECTED
            self.active_regions = self._cluster(recent)

        self._detection_log.append({
            "t"      : t_now,
            "state"  : self.current_state,
            "spikes" : spike_count,
            "regions": len(self.active_regions),
        })

        return self.current_state

    # ── Clustering ─────────────────────────────────────────────────
    def _cluster(self, spikes: list[tuple]) -> list[MotionRegion]:
        """
        Simple greedy spatial clustering of neuron coordinates.
        Groups nearby fired neurons into MotionRegion objects.
        No ML — pure distance thresholding.
        """
        if not spikes:
            return []

        # Convert neuron grid indices to pixel coordinates
        points = np.array([
            [c * self.patch_w + self.patch_w // 2,
             r * self.patch_h + self.patch_h // 2]
            for (r, c, _) in spikes
        ], dtype=np.float32)

        # Greedy cluster: assign each point to nearest existing cluster
        # or start a new one
        cluster_radius_neuron = CLUSTER_RADIUS_PX  # reuse pixel radius
        clusters: list[list[np.ndarray]] = []

        for pt in points:
            assigned = False
            for cl in clusters:
                center = np.mean(cl, axis=0)
                if np.linalg.norm(pt - center) < cluster_radius_neuron:
                    cl.append(pt)
                    assigned = True
                    break
            if not assigned:
                clusters.append([pt])

        regions = []
        for cl in clusters:
            if len(cl) >= CLUSTER_MIN_SPIKES:
                arr = np.array(cl)
                cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
                regions.append(MotionRegion(
                    center_x=float(cx),
                    center_y=float(cy),
                    spike_count=len(cl),
                    label=MOTION_DETECTED
                ))

        return regions

    # ── Accessors ──────────────────────────────────────────────────
    def current_spike_rate(self) -> int:
        return len(self._spike_times)

    def summary(self) -> dict:
        return {
            "state"          : self.current_state,
            "spike_rate"     : self.current_spike_rate(),
            "active_regions" : len(self.active_regions),
        }
