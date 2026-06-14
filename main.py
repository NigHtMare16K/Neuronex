#!/usr/bin/env python3
"""
NeuroStream — Event-Driven Neuromorphic Video Analysis
Hackathon demo entry point.

Pipeline:
  Video → Delta Spike Encoder → LIF + STDP → Time Surface → Pattern Detection
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from input.frame_buffer      import FrameBuffer
from input.video_source      import VideoSource
from input.synthetic_source  import SyntheticVideoSource
from encoding.event_stream   import EventQueue
from encoding.spike_encoder  import SpikeEncoder
from processing.time_surface import TimeSurface
from processing.lif_neuron     import LIFNeuronLayer
from processing.stdp_synapse   import STDPSynapse
from detection.pattern_detector import PatternDetector, STILLNESS
from detection.region_mapper    import RegionMapper
from output.visualizer       import Visualizer
from output.metrics          import MetricsTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NeuroStream — neuromorphic video analysis demo")
    p.add_argument(
        "--source", default="synthetic",
        help="Webcam index (0), video file path, or 'synthetic' (default)")
    p.add_argument(
        "--headless", action="store_true",
        help="Run without GUI (metrics only)")
    p.add_argument(
        "--duration", type=float, default=0,
        help="Auto-stop after N seconds (0 = run until quit)")
    return p.parse_args()


def resolve_source(source: str) -> int | str:
    if source == "synthetic":
        return "synthetic"
    if source.isdigit():
        return int(source)
    return source


def build_pipeline(args: argparse.Namespace):
    shutdown = threading.Event()

    frame_buffer = FrameBuffer()
    event_queue  = EventQueue()
    time_surface = TimeSurface()
    stdp         = STDPSynapse()
    lif          = LIFNeuronLayer(event_queue, time_surface, stdp=stdp)
    detector     = PatternDetector()
    mapper       = RegionMapper()
    metrics      = MetricsTracker()

    visualizer = Visualizer(
        frame_buffer, time_surface, lif, detector, mapper,
        show=not args.headless,
        shutdown=shutdown,
    )

    encoder = SpikeEncoder(
        frame_buffer, event_queue,
        on_spikes=visualizer.register_spikes,
    )

    src = resolve_source(args.source)
    if src == "synthetic":
        video = SyntheticVideoSource(frame_buffer)
    else:
        video = VideoSource(src, frame_buffer)

    return {
        "shutdown": shutdown,
        "video": video,
        "encoder": encoder,
        "lif": lif,
        "detector": detector,
        "visualizer": visualizer,
        "metrics": metrics,
        "stdp": stdp,
    }


def run(args: argparse.Namespace) -> int:
    pipe = build_pipeline(args)
    shutdown  = pipe["shutdown"]
    video     = pipe["video"]
    encoder   = pipe["encoder"]
    lif       = pipe["lif"]
    detector  = pipe["detector"]
    visualizer= pipe["visualizer"]
    metrics   = pipe["metrics"]

    print("\n" + "=" * 52)
    print("  NeuroStream - Event-Driven Neuromorphic Pipeline")
    print("=" * 52)
    print("  Stage 1: Delta spike encoding (artificial retina)")
    print("  Stage 2: LIF neurons + STDP learning")
    print("  Stage 3: Time-surface pattern detection")
    if not args.headless:
        print("  Controls: q=quit  s=spikes  m=membrane  z=zones")
    print("=" * 52 + "\n")

    try:
        video.start()
        encoder.start()
        lif.start()
        metrics.start()
        if not args.headless:
            visualizer.start()

        t_start = time.perf_counter()
        last_frame_count = 0
        last_spike_count = 0
        prev_state = STILLNESS

        while not shutdown.is_set():
            if args.duration > 0 and (time.perf_counter() - t_start) >= args.duration:
                break

            spike_log = lif.drain_spike_log()
            state = detector.update(spike_log)

            enc_stats = encoder.stats()
            new_frames = enc_stats["frames_processed"] - last_frame_count
            if new_frames > 0:
                for _ in range(new_frames):
                    metrics.record_frame()
                last_frame_count = enc_stats["frames_processed"]

            new_spikes = enc_stats["total_spikes"] - last_spike_count
            if new_spikes > 0:
                metrics.record_spikes(new_spikes)
                last_spike_count = enc_stats["total_spikes"]

            if spike_log:
                metrics.record_lif_fires(len(spike_log))
            if state != STILLNESS and prev_state == STILLNESS:
                metrics.record_detection()
            prev_state = state

            if not args.headless:
                s = metrics.snapshot()
                visualizer.update_hud(
                    s["sparsity"], s["mac_savings"],
                    s["lif_fires"], s["detections"])

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[NeuroStream] Interrupted.")
    except RuntimeError as exc:
        print(f"\n[NeuroStream] Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.headless:
            visualizer.stop()
        lif.stop()
        encoder.stop()
        video.stop()
        metrics.stop()

        stdp_stats = pipe["stdp"].stats()
        print(f"\n  STDP weight mean: {stdp_stats['weight_mean']:.3f}  "
              f"(LTP={stdp_stats['potentiation']}, LTD={stdp_stats['depression']})")

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
