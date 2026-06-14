#!/usr/bin/env python3
"""
NeuroStream — Event-Driven Neuromorphic Video Analysis
Hackathon demo entry point.

Modes:
  python main.py                        # synthetic motion (no camera needed)
  python main.py --source 0             # webcam (index 0, 1, 2 ...)
  python main.py --source video.mp4     # process a video file
  python main.py --source video.mp4 --save out.mp4   # save processed output
  python main.py --headless --duration 10             # benchmark only

Pipeline:
  Video/Webcam/Synthetic
    → FrameBuffer (ring buffer)
    → SpikeEncoder (delta filter → sparse ON/OFF events)
    → EventQueue (async)
    → TimeSurface (spatial spike memory)
    → LIFNeuronLayer (integrate-and-fire neurons)
    → STDPSynapse (local weight updates)
    → PatternDetector (rule-based clustering)
    → Visualizer (3-panel live display)
    → MetricsTracker (sparsity / MAC savings)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from input.frame_buffer       import FrameBuffer
from input.video_source       import VideoSource
from input.synthetic_source   import SyntheticVideoSource

from encoding.event_stream    import EventQueue
from encoding.spike_encoder   import SpikeEncoder

from processing.time_surface  import TimeSurface
from processing.lif_neuron    import LIFNeuronLayer
from processing.stdp_synapse  import STDPSynapse

from detection.pattern_detector import PatternDetector, STILLNESS
from detection.region_mapper    import RegionMapper

from output.visualizer    import Visualizer
from output.metrics       import MetricsTracker
from output.alert_handler import AlertHandler


# ── CLI ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NeuroStream — neuromorphic video analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # synthetic demo (no camera)
  python main.py --source 0              # webcam index 0
  python main.py --source 1              # webcam index 1 (try if 0 fails)
  python main.py --source myvideo.mp4    # process a video file
  python main.py --source myvideo.mp4 --save output.mp4
  python main.py --headless --duration 10  # benchmark mode
        """)
    p.add_argument(
        "--source", default="synthetic",
        help="'synthetic' (default), webcam index (0/1/2), or video file path")
    p.add_argument(
        "--headless", action="store_true",
        help="Run without GUI — metrics to stdout only")
    p.add_argument(
        "--duration", type=float, default=0,
        help="Auto-stop after N seconds (0 = run until quit)")
    p.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save processed video to file (e.g. output.mp4)")
    p.add_argument(
        "--threshold", type=int, default=None,
        help="Override spike detection threshold (default from settings)")
    return p.parse_args()


def resolve_source(source: str) -> int | str:
    if source.lower() == "synthetic":
        return "synthetic"
    if source.isdigit():
        return int(source)
    return source   # file path


# ── Pipeline builder ───────────────────────────────────────────────
def build_pipeline(args: argparse.Namespace) -> dict:
    shutdown = threading.Event()

    # Apply CLI overrides to settings
    if args.threshold is not None:
        import config.settings as S
        S.SPIKE_THRESHOLD = args.threshold
        print(f"[Config] Spike threshold overridden → {args.threshold}")

    frame_buffer = FrameBuffer()
    event_queue  = EventQueue()
    time_surface = TimeSurface()
    stdp         = STDPSynapse()
    lif          = LIFNeuronLayer(event_queue, time_surface, stdp=stdp)
    detector     = PatternDetector()
    mapper       = RegionMapper()
    metrics      = MetricsTracker()
    alerts       = AlertHandler(log_path="neurostream_alerts.jsonl")

    visualizer = Visualizer(
        frame_buffer, time_surface, lif, detector, mapper,
        show     = not args.headless,
        shutdown = shutdown,
        save_path= args.save,
    )

    encoder = SpikeEncoder(
        frame_buffer, event_queue,
        on_spikes=visualizer.register_spikes,
    )

    src = resolve_source(args.source)
    if src == "synthetic":
        video = SyntheticVideoSource(frame_buffer)
        print("[NeuroStream] Source → synthetic motion generator")
    elif isinstance(src, int):
        video = VideoSource(src, frame_buffer)
        print(f"[NeuroStream] Source → webcam index {src}")
    else:
        video = VideoSource(src, frame_buffer)
        print(f"[NeuroStream] Source → video file: {src}")

    return {
        "shutdown" : shutdown,
        "video"    : video,
        "encoder"  : encoder,
        "lif"      : lif,
        "stdp"     : stdp,
        "detector" : detector,
        "mapper"   : mapper,
        "visualizer": visualizer,
        "metrics"  : metrics,
        "alerts"   : alerts,
    }


# ── Main run ───────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    pipe     = build_pipeline(args)
    shutdown = pipe["shutdown"]

    _banner(args)

    try:
        # Start pipeline stages in order
        pipe["video"].start()
        pipe["encoder"].start()
        pipe["lif"].start()
        pipe["metrics"].start()
        if not args.headless:
            pipe["visualizer"].start()

        t_start         = time.perf_counter()
        last_frame_count = 0
        last_spike_count = 0
        prev_state       = STILLNESS

        # ── Main coordination loop (20 Hz) ─────────────────────────
        while not shutdown.is_set():
            # Duration limit
            if args.duration > 0:
                if (time.perf_counter() - t_start) >= args.duration:
                    print(f"\n[NeuroStream] Duration {args.duration}s reached.")
                    break

            # Pull LIF output spikes → pattern detector
            spike_log = pipe["lif"].drain_spike_log()
            state     = pipe["detector"].update(spike_log)

            # Update metrics
            enc = pipe["encoder"].stats()

            new_frames = enc["frames_processed"] - last_frame_count
            if new_frames > 0:
                for _ in range(new_frames):
                    pipe["metrics"].record_frame()
                last_frame_count = enc["frames_processed"]

            new_spikes = enc["total_spikes"] - last_spike_count
            if new_spikes > 0:
                pipe["metrics"].record_spikes(new_spikes)
                last_spike_count = enc["total_spikes"]

            if spike_log:
                pipe["metrics"].record_lif_fires(len(spike_log))

            if state != STILLNESS and prev_state == STILLNESS:
                pipe["metrics"].record_detection()

            # Fire alert on state change
            enriched = pipe["mapper"].enrich_regions(pipe["detector"].active_regions)
            pipe["alerts"].update(
                state     = state,
                regions   = enriched,
                spike_rate= pipe["detector"].current_spike_rate(),
            )

            # Push live metrics to HUD
            if not args.headless:
                s = pipe["metrics"].snapshot()
                pipe["visualizer"].update_hud(
                    s["sparsity"], s["mac_savings"],
                    s["lif_fires"], s["detections"]
                )

            prev_state = state
            time.sleep(0.05)   # 20 Hz coordination

    except KeyboardInterrupt:
        print("\n[NeuroStream] Interrupted by user.")

    except RuntimeError as exc:
        print(f"\n[NeuroStream] Fatal error: {exc}", file=sys.stderr)
        return 1

    finally:
        shutdown.set()
        if not args.headless:
            pipe["visualizer"].stop()
        pipe["lif"].stop()
        pipe["encoder"].stop()
        pipe["video"].stop()
        pipe["metrics"].stop()
        pipe["alerts"].close()

        # Final STDP report
        s = pipe["stdp"].stats()
        print(f"\n  STDP  weight_mean={s['weight_mean']:.3f}  "
              f"LTP={s['potentiation']}  LTD={s['depression']}")

        if args.save:
            print(f"  Output saved → {args.save}")

    return 0


def _banner(args: argparse.Namespace) -> None:
    src_label = args.source if args.source != "synthetic" else "synthetic (built-in)"
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       NeuroStream — Neuromorphic Pipeline            ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Source   : {src_label:<41}║")
    print(f"║  Headless : {'Yes' if args.headless else 'No':<41}║")
    print(f"║  Duration : {str(args.duration)+'s' if args.duration else 'Until quit':<41}║")
    print(f"║  Save     : {str(args.save) if args.save else 'No':<41}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Stage 1: Delta spike encoding  (artificial retina) ║")
    print("║  Stage 2: LIF neurons + STDP learning               ║")
    print("║  Stage 3: Time-surface pattern detection            ║")
    if not args.headless:
        print("║  Controls: q=quit  s=spikes  m=membrane  z=zones   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
