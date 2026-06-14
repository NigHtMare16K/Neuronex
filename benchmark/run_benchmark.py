#!/usr/bin/env python3
"""
run_benchmark.py
Runs NeuroStream headless vs dense OpenCV baseline
and prints a side-by-side comparison table.

Usage:
  python benchmark/run_benchmark.py --duration 10
  python benchmark/run_benchmark.py --duration 10 --source myvideo.mp4
"""

import sys
import os
import time
import argparse
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.baseline_opencv import BaselineOpenCV
from encoding.spike_encoder    import SpikeEncoder
from encoding.event_stream     import EventQueue
from input.frame_buffer        import FrameBuffer
from input.synthetic_source    import SyntheticVideoSource
from input.video_source        import VideoSource
from processing.time_surface   import TimeSurface
from processing.lif_neuron     import LIFNeuronLayer
from processing.stdp_synapse   import STDPSynapse
from config.settings           import FRAME_WIDTH, FRAME_HEIGHT


def parse_args():
    p = argparse.ArgumentParser(description="NeuroStream benchmark")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Benchmark duration in seconds (default 10)")
    p.add_argument("--source", default="synthetic",
                   help="'synthetic', webcam index, or video file")
    return p.parse_args()


def run_neurostream(source, duration: float) -> dict:
    """Run neuromorphic pipeline headless and collect metrics."""
    frame_buffer = FrameBuffer()
    event_queue  = EventQueue()
    time_surface = TimeSurface()
    stdp         = STDPSynapse()
    lif          = LIFNeuronLayer(event_queue, time_surface, stdp=stdp)

    encoder = SpikeEncoder(frame_buffer, event_queue)

    if source == "synthetic" or source == 0 or (isinstance(source, str) and not source.isdigit()):
        video = SyntheticVideoSource(frame_buffer) if source == "synthetic" else VideoSource(source, frame_buffer)
    else:
        video = VideoSource(int(source), frame_buffer)

    video.start()
    encoder.start()
    lif.start()

    time.sleep(duration)

    lif.stop()
    encoder.stop()
    video.stop()

    stats = encoder.stats()
    total_pixels   = stats["frames_processed"] * FRAME_WIDTH * FRAME_HEIGHT
    total_spikes   = stats["total_spikes"]
    sparsity       = 1.0 - (total_spikes / max(total_pixels, 1))
    mac_savings    = sparsity
    lif_fires      = lif.total_output_spikes()

    return {
        "frames"        : stats["frames_processed"],
        "total_spikes"  : total_spikes,
        "sparsity_pct"  : sparsity * 100,
        "mac_savings_pct": mac_savings * 100,
        "lif_fires"     : lif_fires,
        "neuro_macs"    : total_spikes,                        # only spikes cost MACs
        "dense_macs"    : total_pixels,                        # baseline: every pixel
    }


def run_baseline(source, duration: float) -> dict:
    """Run dense OpenCV pipeline and collect metrics."""
    frame_buffer = FrameBuffer()
    baseline     = BaselineOpenCV()

    if source == "synthetic":
        video = SyntheticVideoSource(frame_buffer)
    else:
        video = VideoSource(source if isinstance(source, int) else
                            (int(source) if str(source).isdigit() else source),
                            frame_buffer)

    video.start()
    t0 = time.perf_counter()
    frames_done = 0

    while time.perf_counter() - t0 < duration:
        frame = frame_buffer.get(timeout=0.1)
        if frame is not None:
            baseline.process_frame(frame)
            frames_done += 1

    video.stop()
    return baseline.stats()


def print_report(neuro: dict, dense: dict, duration: float) -> None:
    speedup  = dense["avg_latency_ms"] / max(neuro["neuro_macs"] /
               max(neuro["frames"], 1) / 1000, 0.001)
    mac_ratio = neuro["neuro_macs"] / max(dense["total_macs"], 1)

    sep = "─" * 56
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          NeuroStream Benchmark Results               ║")
    print(f"║          Duration: {duration:.0f}s                              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  {'Metric':<28} {'NeuroStream':>10} {'OpenCV':>10}  ║")
    print("╠" + sep + "╣")
    print(f"║  {'Frames processed':<28} {neuro['frames']:>10} {dense['frames']:>10}  ║")
    print(f"║  {'Total MACs':<28} {neuro['neuro_macs']:>10,} {dense['total_macs']:>10,}  ║")
    print(f"║  {'MACs per frame':<28} {neuro['neuro_macs']//max(neuro['frames'],1):>10,} {dense['macs_per_frame']:>10,}  ║")
    print(f"║  {'Sparsity':<28} {neuro['sparsity_pct']:>9.1f}% {'N/A':>10}  ║")
    print(f"║  {'MAC savings vs baseline':<28} {neuro['mac_savings_pct']:>9.1f}% {'0.0%':>10}  ║")
    print(f"║  {'LIF neuron fires':<28} {neuro['lif_fires']:>10} {'N/A':>10}  ║")
    print(f"║  {'Avg latency ms/frame':<28} {'sparse':>10} {dense['avg_latency_ms']:>9.2f}  ║")
    print("╠" + sep + "╣")
    print(f"║  MAC reduction ratio  : {mac_ratio:.4f}x  ({(1-mac_ratio)*100:.1f}% fewer operations)  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Interpretation:")
    print(f"  NeuroStream used {neuro['mac_savings_pct']:.1f}% fewer MAC operations")
    print(f"  than dense frame-by-frame processing — achieved through")
    print(f"  spike sparsity alone, without any ML model or GPU.\n")


def main():
    args   = parse_args()
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"\n[Benchmark] Running {args.duration}s neuromorphic pipeline...")
    neuro = run_neurostream(source, args.duration)

    print(f"[Benchmark] Running {args.duration}s dense OpenCV baseline...")
    dense = run_baseline(source, args.duration)

    print_report(neuro, dense, args.duration)


if __name__ == "__main__":
    main()