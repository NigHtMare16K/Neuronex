"""
metrics.py
Tracks and reports pipeline efficiency metrics.

Power proxy model:
  - Each spike = 1 MAC operation (sparse)
  - Baseline (dense CNN) = FRAME_WIDTH * FRAME_HEIGHT MACs per frame
  - Power savings = 1 - (spike_MACs / baseline_MACs)

We do NOT claim actual watt measurements —
these are compute-operation proxies, which is the standard
approach in neuromorphic benchmarking literature (e.g. Intel Loihi papers).
"""

import time
import threading
from collections import deque
from config.settings import (
    METRICS_LOG_INTERVAL_S, BASELINE_MAC_PER_FRAME, TARGET_FPS
)


class MetricsTracker:
    """
    Accumulates stats from all pipeline stages and periodically
    prints a summary table to stdout.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Counters
        self._frames_total   = 0
        self._spikes_total   = 0
        self._lif_fires_total= 0
        self._detections     = 0

        # Rolling window for rate calculations (last 2 seconds)
        self._spike_window   : deque[tuple[float, int]] = deque()
        self._fire_window    : deque[tuple[float, int]] = deque()

        # Timing
        self._start_time     = time.perf_counter()
        self._last_report_t  = time.perf_counter()

        # Background reporter thread
        self._running = False
        self._thread  = threading.Thread(target=self._report_loop,
                                         daemon=True, name="Metrics")

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.print_summary()

    # ── Record ─────────────────────────────────────────────────────
    def record_frame(self) -> None:
        with self._lock:
            self._frames_total += 1

    def record_spikes(self, count: int) -> None:
        t = time.perf_counter()
        with self._lock:
            self._spikes_total += count
            self._spike_window.append((t, count))
            # Prune old entries
            cutoff = t - 2.0
            while self._spike_window and self._spike_window[0][0] < cutoff:
                self._spike_window.popleft()

    def record_lif_fires(self, count: int) -> None:
        t = time.perf_counter()
        with self._lock:
            self._lif_fires_total += count
            self._fire_window.append((t, count))
            cutoff = t - 2.0
            while self._fire_window and self._fire_window[0][0] < cutoff:
                self._fire_window.popleft()

    def record_detection(self) -> None:
        with self._lock:
            self._detections += 1

    # ── Compute ────────────────────────────────────────────────────
    def sparsity(self) -> float:
        """Fraction of pixels that did NOT spike across all frames."""
        with self._lock:
            total_pixels = self._frames_total * BASELINE_MAC_PER_FRAME
            if total_pixels == 0:
                return 0.0
            return max(0.0, 1.0 - self._spikes_total / total_pixels)

    def mac_savings(self) -> float:
        """Fraction of baseline MACs avoided (power proxy)."""
        return self.sparsity()

    def spike_rate_hz(self) -> float:
        """Spikes per second over the last 2s window."""
        with self._lock:
            if not self._spike_window:
                return 0.0
            total = sum(c for _, c in self._spike_window)
            window_s = 2.0
            return total / window_s

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start_time

    # ── Reporting ──────────────────────────────────────────────────
    def _report_loop(self) -> None:
        while self._running:
            time.sleep(METRICS_LOG_INTERVAL_S)
            self._print_live()

    def _print_live(self) -> None:
        with self._lock:
            frames  = self._frames_total
            spikes  = self._spikes_total
            fires   = self._lif_fires_total
            detects = self._detections

        spar  = self.sparsity()
        saved = self.mac_savings()
        rate  = self.spike_rate_hz()
        fps   = frames / max(self.elapsed_s(), 0.001)

        print("\n" + "-" * 52)
        print(f"  NeuroStream Metrics  [{self.elapsed_s():.1f}s]")
        print("-" * 52)
        print(f"  Frames processed  : {frames}")
        print(f"  Pipeline FPS      : {fps:.1f}")
        print(f"  Total spikes      : {spikes:,}")
        print(f"  Spike rate        : {rate:.0f} spikes/s")
        print(f"  LIF fires         : {fires:,}")
        print(f"  Detections        : {detects}")
        print(f"  Sparsity          : {spar * 100:.1f}%")
        print(f"  MAC savings       : {saved * 100:.1f}%  (vs dense baseline)")
        print("-" * 52)

    def print_summary(self) -> None:
        print("\n" + "=" * 52)
        print("  FINAL SUMMARY")
        self._print_live()
        print("=" * 52)

    def snapshot(self) -> dict:
        """Return current metrics as a dict (for alert_handler etc.)."""
        return {
            "frames"     : self._frames_total,
            "spikes"     : self._spikes_total,
            "lif_fires"  : self._lif_fires_total,
            "sparsity"   : self.sparsity(),
            "mac_savings": self.mac_savings(),
            "detections" : self._detections,
            "elapsed_s"  : self.elapsed_s(),
        }
