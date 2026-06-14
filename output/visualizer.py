"""
visualizer.py
Thread 4 — Renders the live NeuroStream display.

Three panels side by side:
  [1] Original grayscale frame
  [2] Spike event map  (ON=green, OFF=red dots)
  [3] LIF membrane potential heatmap + detection overlays

HUD overlay shows live sparsity %, MAC savings %, LIF fires, detections.

Press 'q' to quit, 's' spikes, 'm' membrane, 'z' zones.
Supports shutdown Event for clean thread coordination.
"""

import time
import threading
import numpy as np
import cv2

from processing.time_surface    import TimeSurface
from processing.lif_neuron      import LIFNeuronLayer
from detection.pattern_detector import PatternDetector, STILLNESS, HIGH_ACTIVITY
from detection.region_mapper    import RegionMapper
from input.frame_buffer         import FrameBuffer
from encoding.event_stream      import SpikeEvent
from config.settings import (
    FRAME_WIDTH, FRAME_HEIGHT,
    NEURON_GRID_ROWS, NEURON_GRID_COLS
)

# ── Colours (BGR) ──────────────────────────────────────────────────
COL_ON      = (0,   255,  0)
COL_OFF     = (0,   0,   255)
COL_BOX     = (0,   165, 255)
COL_STILL   = (180, 180, 180)
COL_MOTION  = (0,   200, 255)
COL_HIGH    = (0,   0,   255)
COL_HUD_BG  = (20,  20,  20)
COL_WHITE   = (255, 255, 255)
COL_GREEN   = (100, 255, 100)
COL_YELLOW  = (0,   220, 255)


class Visualizer:
    def __init__(self,
                 frame_buffer : FrameBuffer,
                 time_surface : TimeSurface,
                 lif_layer    : LIFNeuronLayer,
                 detector     : PatternDetector,
                 region_mapper: RegionMapper,
                 show         : bool = True,
                 shutdown     : threading.Event | None = None,
                 save_path    : str | None = None):

        self.frame_buffer  = frame_buffer
        self.time_surface  = time_surface
        self.lif_layer     = lif_layer
        self.detector      = detector
        self.mapper        = region_mapper
        self.show          = show
        self._shutdown     = shutdown or threading.Event()
        self._save_path    = save_path

        # Toggle flags
        self._show_spikes   = True
        self._show_membrane = True
        self._show_zones    = False

        # HUD state (updated by main loop via update_hud)
        self._hud = {
            "sparsity"   : 0.0,
            "mac_savings": 0.0,
            "lif_fires"  : 0,
            "detections" : 0,
        }
        self._hud_lock = threading.Lock()

        # Spike buffer for overlay
        self._spike_buf_lock = threading.Lock()
        self._spike_buf: list[SpikeEvent] = []

        # Video writer for file output
        self._writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                save_path, fourcc, 20,
                (FRAME_WIDTH * 3, FRAME_HEIGHT)   # 3 panels wide
            )

        self._running = False
        self._thread  = threading.Thread(target=self._render_loop,
                                         daemon=True, name="Visualizer")
        self._fps_counter = 0
        self._last_fps_t  = time.perf_counter()
        self.fps = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._thread.start()
        print("[Visualizer] Started.")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
        print("[Visualizer] Stopped.")

    # ── Public API ─────────────────────────────────────────────────
    def register_spikes(self, events: list[SpikeEvent]) -> None:
        """Called by SpikeEncoder after each frame encode."""
        with self._spike_buf_lock:
            self._spike_buf = events[-600:] if len(events) > 600 else list(events)

    def update_hud(self, sparsity: float, mac_savings: float,
                   lif_fires: int, detections: int) -> None:
        """Called by main loop to push live metrics to the HUD."""
        with self._hud_lock:
            self._hud["sparsity"]    = sparsity
            self._hud["mac_savings"] = mac_savings
            self._hud["lif_fires"]   = lif_fires
            self._hud["detections"]  = detections

    # ── Render loop ────────────────────────────────────────────────
    def _render_loop(self) -> None:
        while self._running:
            frame = self.frame_buffer.peek_latest()
            if frame is None:
                time.sleep(0.02)
                continue

            canvas = self._compose(frame)

            if self.show:
                cv2.imshow("NeuroStream — Neuromorphic Pipeline", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._running = False
                    self._shutdown.set()
                elif key == ord('s'):
                    self._show_spikes   = not self._show_spikes
                elif key == ord('m'):
                    self._show_membrane = not self._show_membrane
                elif key == ord('z'):
                    self._show_zones    = not self._show_zones

            if self._writer:
                self._writer.write(canvas)

            self._fps_counter += 1
            now = time.perf_counter()
            if now - self._last_fps_t >= 1.0:
                self.fps = self._fps_counter / (now - self._last_fps_t)
                self._fps_counter = 0
                self._last_fps_t  = now

            time.sleep(0.016)

    # ── Composition ────────────────────────────────────────────────
    def _compose(self, gray_frame: np.ndarray) -> np.ndarray:
        p1 = self._input_panel(gray_frame)
        p2 = self._spike_panel()
        p3 = self._membrane_panel()

        for img, lbl in [(p1, "INPUT FRAME"), (p2, "SPIKE MAP"), (p3, "LIF + DETECTIONS")]:
            cv2.putText(img, lbl, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)

        canvas = np.hstack([p1, p2, p3])
        self._draw_hud(canvas)
        return canvas

    def _input_panel(self, gray: np.ndarray) -> np.ndarray:
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # FPS bottom-left
        cv2.putText(panel, f"FPS {self.fps:.1f}",
                    (8, FRAME_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GREEN, 1, cv2.LINE_AA)
        return panel

    def _spike_panel(self) -> np.ndarray:
        panel = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        with self._spike_buf_lock:
            spikes = list(self._spike_buf)

        if self._show_spikes:
            for ev in spikes:
                col = COL_ON if ev.is_on() else COL_OFF
                cv2.circle(panel, (ev.x, ev.y), 2, col, -1)

        # Legend
        cv2.circle(panel, (12, FRAME_HEIGHT - 28), 4, COL_ON,  -1)
        cv2.putText(panel, "ON",  (20, FRAME_HEIGHT - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ON,  1)
        cv2.circle(panel, (50, FRAME_HEIGHT - 28), 4, COL_OFF, -1)
        cv2.putText(panel, "OFF", (58, FRAME_HEIGHT - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_OFF, 1)
        cv2.putText(panel, f"Events: {len(spikes)}",
                    (8, FRAME_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_WHITE, 1, cv2.LINE_AA)
        return panel

    def _membrane_panel(self) -> np.ndarray:
        mem      = self.lif_layer.get_membrane_map()
        mem_norm = cv2.normalize(mem, None, 0, 255, cv2.NORM_MINMAX)
        mem_up   = cv2.resize(mem_norm.astype(np.uint8),
                              (FRAME_WIDTH, FRAME_HEIGHT),
                              interpolation=cv2.INTER_NEAREST)
        panel = cv2.applyColorMap(mem_up, cv2.COLORMAP_INFERNO)

        # Draw detection regions
        regions = self.mapper.enrich_regions(self.detector.active_regions)
        for r in regions:
            x0, y0, x1, y1 = r["bbox"]
            cv2.rectangle(panel, (x0, y0), (x1, y1), COL_BOX, 2)
            cv2.putText(panel, r["zone"], (x0 + 4, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_BOX, 1, cv2.LINE_AA)
            # Spike count inside box
            cv2.putText(panel, f"n={r['spike_count']}",
                        (x0 + 4, y0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_WHITE, 1, cv2.LINE_AA)

        # Zone grid overlay
        if self._show_zones:
            for zone in self.mapper.zone_grid():
                zx0, zy0, zx1, zy1 = zone["bbox"]
                cv2.rectangle(panel, (zx0, zy0), (zx1, zy1), (80, 80, 80), 1)
                cx, cy = (zx0 + zx1) // 2, (zy0 + zy1) // 2
                cv2.putText(panel, zone["name"], (cx - 25, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # State label
        state = self.detector.current_state
        col   = (COL_STILL if state == STILLNESS else
                 COL_HIGH  if state == HIGH_ACTIVITY else COL_MOTION)
        cv2.putText(panel, state, (8, FRAME_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return panel

    def _draw_hud(self, canvas: np.ndarray) -> None:
        """Draw semi-transparent HUD bar at top of full canvas."""
        with self._hud_lock:
            hud = dict(self._hud)

        bar_h = 28
        overlay = canvas[:bar_h, :].copy()
        canvas[:bar_h, :] = (overlay * 0.35).astype(np.uint8)

        items = [
            f"Sparsity: {hud['sparsity']*100:.1f}%",
            f"MAC saved: {hud['mac_savings']*100:.1f}%",
            f"LIF fires: {hud['lif_fires']}",
            f"Detections: {hud['detections']}",
            "q=quit  s=spikes  m=membrane  z=zones",
        ]
        x = 10
        for item in items:
            cv2.putText(canvas, item, (x, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_YELLOW, 1, cv2.LINE_AA)
            x += len(item) * 8 + 20