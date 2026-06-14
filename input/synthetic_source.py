"""
synthetic_source.py
Generates synthetic motion frames — no webcam or file needed.
Used for demo/testing when --source synthetic (default).
 
Produces:
  - Moving bright blobs (simulates walking figures)
  - Scrolling edge patterns (simulates scene changes)
  - Random noise bursts (simulates high-activity events)
 
This ensures the demo ALWAYS works even without a camera.
"""
 
import time
import math
import threading
import numpy as np
 
from input.frame_buffer import FrameBuffer
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
 
 
class SyntheticVideoSource:
    """
    Generates synthetic grayscale frames and pushes them into FrameBuffer.
    Mimics real motion patterns: blobs moving across the frame,
    edge flickers, and occasional burst events.
    """
 
    def __init__(self, buffer: FrameBuffer):
        self.buffer   = buffer
        self._running = False
        self._thread  = threading.Thread(target=self._generate_loop,
                                         daemon=True, name="SyntheticSource")
        self._frame_interval = 1.0 / TARGET_FPS
        self._t0 = 0.0
 
        # Blob state: list of dicts with pos, vel, radius, intensity
        self._blobs = [
            {"x": 100.0, "y": 100.0, "vx":  1.8, "vy":  0.9, "r": 40, "i": 200},
            {"x": 400.0, "y": 200.0, "vx": -1.2, "vy":  1.5, "r": 30, "i": 180},
            {"x": 200.0, "y": 280.0, "vx":  2.1, "vy": -1.1, "r": 25, "i": 220},
        ]
 
    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._t0 = time.perf_counter()
        self._thread.start()
        print("[SyntheticSource] Started — generating synthetic motion frames.")
 
    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        print("[SyntheticSource] Stopped.")
 
    # ── Generation loop ────────────────────────────────────────────
    def _generate_loop(self) -> None:
        last_time = time.perf_counter()
 
        while self._running:
            t = time.perf_counter() - self._t0
            frame = self._make_frame(t)
            self.buffer.put(frame)
 
            elapsed = time.perf_counter() - last_time
            sleep_t = self._frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_time = time.perf_counter()
 
    def _make_frame(self, t: float) -> np.ndarray:
        """
        Render one synthetic grayscale frame at time t (seconds).
        Layers:
          1. Dark background with slow gradient
          2. Moving blobs (Gaussian)
          3. Sinusoidal edge bar (simulates scene panning)
          4. Occasional noise burst
        """
        W, H = FRAME_WIDTH, FRAME_HEIGHT
        frame = np.zeros((H, W), dtype=np.float32)
 
        # 1. Background gradient (slow pulse)
        bg = 20 + 10 * math.sin(t * 0.3)
        frame[:] = bg
 
        # 2. Sinusoidal edge bar (horizontal scroll)
        bar_x = int((math.sin(t * 0.4) * 0.5 + 0.5) * W)
        x_grid = np.arange(W, dtype=np.float32)
        bar = 60 * np.exp(-0.5 * ((x_grid - bar_x) / 15) ** 2)
        frame += bar[np.newaxis, :]
 
        # 3. Moving Gaussian blobs
        y_grid = np.arange(H, dtype=np.float32)[:, np.newaxis]
        x_grid = np.arange(W, dtype=np.float32)[np.newaxis, :]
 
        for blob in self._blobs:
            # Update position
            blob["x"] += blob["vx"]
            blob["y"] += blob["vy"]
            # Bounce off walls
            if blob["x"] < blob["r"] or blob["x"] > W - blob["r"]:
                blob["vx"] *= -1
            if blob["y"] < blob["r"] or blob["y"] > H - blob["r"]:
                blob["vy"] *= -1
 
            # Add Gaussian blob
            dist2 = (x_grid - blob["x"]) ** 2 + (y_grid - blob["y"]) ** 2
            sigma2 = (blob["r"] / 2.0) ** 2
            frame += blob["i"] * np.exp(-dist2 / (2 * sigma2))
 
        # 4. Noise burst every ~5 seconds
        burst_phase = t % 5.0
        if burst_phase < 0.15:
            noise_strength = 40 * (1 - burst_phase / 0.15)
            frame += np.random.uniform(0, noise_strength, (H, W)).astype(np.float32)
 
        return np.clip(frame, 0, 255).astype(np.uint8)
 
    @property
    def is_running(self) -> bool:
        return self._running