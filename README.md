# NeuroStream

**Event-driven neuromorphic video analysis** — processes only what changes, when it changes.

Traditional video pipelines run dense convolutions on every pixel every frame. NeuroStream mimics biological retinas: a delta encoder converts frames into sparse spike events, LIF neurons with STDP learn local patterns without backprop, and time-surface maps enable motion detection at a fraction of the compute cost.

## Quick start

```bash
pip install -r requirements.txt
python main.py                  # synthetic motion demo (no camera needed)
python main.py --source 0         # webcam
python main.py --source video.mp4 # video file
```

Press **q** to quit. Toggle overlays: **s** spikes, **m** membrane, **z** zones.

## Architecture

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `SpikeEncoder` | Frame → sparse ON/OFF events (95%+ data reduction) |
| 2 | `LIFNeuronLayer` + `STDPSynapse` | Event-driven neurons, causal Hebbian learning |
| 3 | `TimeSurface` + `PatternDetector` | Spatial memory surfaces → motion regions |

## Benchmark

```bash
python benchmark/run_benchmark.py --duration 10
```

Prints sparsity % and MAC savings vs a dense per-frame baseline.

## Hackathon pitch points

- **95%+ sparsity** — most pixels never fire; MAC units stay idle (power proxy)
- **No backprop** — STDP is fully local, biologically plausible
- **Async pipeline** — capture, encode, process, render on separate threads
- **Edge-ready** — 320×240 @ 30 FPS on CPU; designed for Pi / Loihi class hardware
