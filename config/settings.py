"""
NeuroStream — central configuration.
Tuned for real-time demo on laptop / Raspberry Pi class hardware.
"""

# ── Frame / capture ────────────────────────────────────────────────
FRAME_WIDTH        = 320
FRAME_HEIGHT       = 240
TARGET_FPS         = 30
FRAME_BUFFER_SIZE  = 4

# ── Event encoding (Stage 1 — artificial retina) ───────────────────
SPIKE_THRESHOLD    = 15          # luminance delta θ (0–255 scale)
SPIKE_SUBSAMPLE    = 2           # pixel stride — cuts encode load ~4x
MAX_SPIKES_FRAME   = 3000        # cap events per frame for real-time
POLARITY_ON        = 1
POLARITY_OFF       = -1
EVENT_QUEUE_MAXSIZE = 50_000

# ── LIF neuron grid (Stage 2) ──────────────────────────────────────
NEURON_GRID_ROWS   = 15
NEURON_GRID_COLS   = 20
LEAK_FACTOR        = 0.92        # membrane leak per 5 ms tick
V_REST             = 0.0
V_THRESH           = 1.8
V_RESET            = 0.0
REFRACTORY_MS      = 20.0

# ── STDP synapses ──────────────────────────────────────────────────
STDP_A_PLUS        = 0.05
STDP_A_MINUS       = 0.04
STDP_TAU           = 20.0        # ms
WEIGHT_MIN         = 0.1
WEIGHT_MAX         = 1.0

# ── Time surface (Stage 3) ─────────────────────────────────────────
TIME_SURFACE_TAU   = 50.0        # ms decay constant

# ── Pattern detection ──────────────────────────────────────────────
SPIKE_RATE_WINDOW_MS    = 500
MOTION_SPIKE_THRESHOLD  = 8
CLUSTER_MIN_SPIKES      = 3
CLUSTER_RADIUS_PX       = 40

# ── Metrics / benchmarking ─────────────────────────────────────────
METRICS_LOG_INTERVAL_S  = 3.0
BASELINE_MAC_PER_FRAME  = FRAME_WIDTH * FRAME_HEIGHT   # dense CNN proxy

# ── Display ────────────────────────────────────────────────────────
DISPLAY_SCALE      = 0.85        # shrink window for smoother FPS
SPIKE_OVERLAY_MAX  = 600         # max dots drawn per panel
