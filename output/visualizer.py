"""
visualizer.py
Thread 4 — Renders the live NeuroStream display.

Three panels side by side:
  [1] Original grayscale frame
  [2] Spike event map  (ON=green, OFF=red dots)
  [3] LIF membrane potential heatmap + detection overlays

Press 'q' to quit, 's' to toggle spike overlay,
      'm' to toggle membrane map, 'z' to toggle zone grid.
"""

import time
import threading
import numpy as np
import cv2

from processing.time_surface   import TimeSurface
from processing.lif_neuron     import LIFNeuronLayer
from detection.pattern_detector import PatternDetector, STILLNESS, HIGH_ACTIVITY
from detection.region_mapper   import RegionMapper
from input.frame_buffer        import FrameBuffer
from encoding.event_stream     import EventQueue, SpikeEvent
from config.settings import (
    FRAME_WIDTH, FRAME_HEIGHT,
    DISPLAY_SCALE, SPIKE_OVERLAY_MAX,
)

# ── Colours (BGR) ──────────────────────────────────────────────────
COL_ON      = (0,   255, 0)    # Green  — ON spike
COL_OFF     = (0,   0,   255)  # Red    — OFF spike
COL_FIRE    = (0,   255, 255)  # Yellow — LIF fired
COL_BOX     = (255, 165, 0)    # Orange — motion bbox
COL_STILL   = (200, 200, 200)  # Grey   — stillness label
COL_MOTION  = (0,   200, 255)  # Cyan   — motion label
COL_HIGH    = (0,   0,   255)  # Red    — high activity label


class Visualizer:
    """
    Composites all pipeline layers into a single display window.
    Optionally headless (show=False) for benchmarking runs.
    """

    def __init__(self,
                 frame_buffer   : FrameBuffer,
                 time_surface   : TimeSurface,
                 lif_layer      : LIFNeuronLayer,
                 detector       : PatternDetector,
                 region_mapper  : RegionMapper,
                 show           : bool = True,
                 shutdown       : threading.Event | None = None):

        self.frame_buffer  = frame_buffer
        self.time_surface  = time_surface
        self.lif_layer     = lif_layer
        self.detector      = detector
        self.mapper        = region_mapper
        self.show          = show
        self._shutdown     = shutdown

        # Toggle flags
        self._show_spikes   = True
        self._show_membrane = True
        self._show_zones    = False

        # Spike event buffer for overlay (filled by register_spike)
        self._spike_buf_lock = threading.Lock()
        self._spike_buf      : list[SpikeEvent] = []

        self._running = False
        self._thread  = threading.Thread(target=self._render_loop,
                                         daemon=True, name="Visualizer")
        self._fps_counter = 0
        self._last_fps_t  = time.perf_counter()
        self.fps = 0.0

        # Presentation HUD (updated from main loop)
        self._hud_lock = threading.Lock()
        self._hud = {
            "sparsity": 0.0, "mac_saved": 0.0,
            "lif_fires": 0, "detections": 0,
        }

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._thread.start()
        print("[Visualizer] Started.")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("[Visualizer] Stopped.")

    def register_spikes(self, events: list[SpikeEvent]) -> None:
        """Called by encoder to push latest spikes for overlay."""
        cap = SPIKE_OVERLAY_MAX
        with self._spike_buf_lock:
            self._spike_buf = events[-cap:] if len(events) > cap else events

    def update_hud(self, sparsity: float, mac_saved: float,
                   lif_fires: int, detections: int) -> None:
        with self._hud_lock:
            self._hud = {
                "sparsity": sparsity, "mac_saved": mac_saved,
                "lif_fires": lif_fires, "detections": detections,
            }

    # ── Render loop ────────────────────────────────────────────────
    def _render_loop(self) -> None:
        while self._running:
            frame = self.frame_buffer.peek_latest()
            if frame is None:
                time.sleep(0.02)
                continue

            canvas = self._compose(frame)

            if self.show:
                cv2.imshow("NeuroStream", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._running = False
                    if self._shutdown:
                        self._shutdown.set()
                elif key == ord('s'):
                    self._show_spikes   = not self._show_spikes
                elif key == ord('m'):
                    self._show_membrane = not self._show_membrane
                elif key == ord('z'):
                    self._show_zones    = not self._show_zones

            self._fps_counter += 1
            now = time.perf_counter()
            if now - self._last_fps_t >= 1.0:
                self.fps = self._fps_counter / (now - self._last_fps_t)
                self._fps_counter = 0
                self._last_fps_t  = now

            time.sleep(0.008)

    # ── Composition ────────────────────────────────────────────────
    def _compose(self, gray_frame: np.ndarray) -> np.ndarray:
        h, w = FRAME_HEIGHT, FRAME_WIDTH

        # Panel 1 — original grayscale
        p1 = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Panel 2 — spike event map
        p2 = self._spike_panel()

        # Panel 3 — membrane heatmap + detections
        p3 = self._membrane_panel(gray_frame)

        # Annotate each panel with stage labels
        for img, lbl in [
            (p1, "STAGE 1: INPUT"),
            (p2, "STAGE 2: SPIKE EVENTS"),
            (p3, "STAGE 3: LIF + DETECT"),
        ]:
            cv2.putText(img, lbl, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(p1, f"Render: {self.fps:.0f} fps", (8, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

        canvas = np.hstack([p1, p2, p3])
        canvas = self._draw_hud(canvas)
        if DISPLAY_SCALE != 1.0:
            nw = int(canvas.shape[1] * DISPLAY_SCALE)
            nh = int(canvas.shape[0] * DISPLAY_SCALE)
            canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas

    def _draw_hud(self, canvas: np.ndarray) -> np.ndarray:
        """Title + live stats bar for demo presentation."""
        w = canvas.shape[1]
        with self._hud_lock:
            hud = self._hud.copy()
        state = self.detector.current_state

        cv2.rectangle(canvas, (0, 0), (w, 32), (25, 25, 35), -1)
        cv2.putText(canvas, "NeuroStream  |  Event-Driven Neuromorphic Vision",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)

        bar_y = canvas.shape[0] - 28
        cv2.rectangle(canvas, (0, bar_y), (w, canvas.shape[0]), (25, 25, 35), -1)
        stats = (
            f"Sparsity: {hud['sparsity']*100:.1f}%   "
            f"MAC saved: {hud['mac_saved']*100:.1f}%   "
            f"State: {state}   "
            f"LIF fires: {hud['lif_fires']}   "
            f"Detections: {hud['detections']}"
        )
        cv2.putText(canvas, stats, (10, bar_y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return canvas

    def _spike_panel(self) -> np.ndarray:
        panel = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        if self._show_spikes:
            with self._spike_buf_lock:
                spikes = self._spike_buf
            for ev in spikes:
                if 0 <= ev.y < FRAME_HEIGHT and 0 <= ev.x < FRAME_WIDTH:
                    panel[ev.y, ev.x] = COL_ON if ev.is_on() else COL_OFF

        n = len(self._spike_buf)
        cv2.putText(panel, f"Events: {n}", (8, FRAME_HEIGHT - 10),
                    (8, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return panel

    def _membrane_panel(self, gray_frame: np.ndarray) -> np.ndarray:
        mem = self.lif_layer.get_membrane_map()        # (rows, cols)

        # Resize membrane grid to frame size
        mem_norm = cv2.normalize(mem, None, 0, 255, cv2.NORM_MINMAX)
        mem_up   = cv2.resize(mem_norm.astype(np.uint8),
                              (FRAME_WIDTH, FRAME_HEIGHT),
                              interpolation=cv2.INTER_NEAREST)
        panel = cv2.applyColorMap(mem_up, cv2.COLORMAP_HOT)

        # Draw detection regions
        regions = self.mapper.enrich_regions(self.detector.active_regions)
        for r in regions:
            x0, y0, x1, y1 = r["bbox"]
            cv2.rectangle(panel, (x0, y0), (x1, y1), COL_BOX, 2)
            cv2.putText(panel, r["zone"], (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_BOX, 1)

        # State label
        state = self.detector.current_state
        col   = COL_STILL if state == STILLNESS else (
                COL_HIGH  if state == HIGH_ACTIVITY else COL_MOTION)
        cv2.putText(panel, state, (8, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        return panel
