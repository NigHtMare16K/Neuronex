# 3-Minute NeuroStream Presentation

## Before you start (30 sec setup)

```bash
pip install -r requirements.txt
python main.py
```

Use **synthetic** mode (default) — reliable motion, no webcam issues.  
Full-screen the window. Point to the **title bar** and **bottom stats bar**.

---

## Script (~3 minutes)

### 1. Problem (30 sec)
> "Normal video AI runs convolutions on **every pixel, every frame** — even static backgrounds. That's 90%+ wasted compute and power."

### 2. Insight (20 sec)
> "Biological retinas don't send frames — they send **spikes only when something changes**. We built NeuroStream to do the same."

### 3. Live demo — walk left to right (90 sec)

| Panel | Say this |
|-------|----------|
| **Stage 1** | "Standard video input — 320×240 at 30 fps." |
| **Stage 2** | "Our delta encoder fires ON/OFF spikes only where luminance changes. Watch — **most pixels stay black**. That's ~95% sparsity." |
| **Stage 3** | "Spikes feed LIF neurons with STDP learning — no backprop. Orange boxes = motion detected with zone labels." |
| **Bottom bar** | "Live stats: sparsity %, MAC savings vs dense CNN, detection state." |

Pause when blobs move — point out spikes appear **only on edges**, not the whole frame.

### 4. Why it wins (40 sec)
- **Event-driven** — compute scales with motion, not resolution
- **STDP** — local Hebbian learning, edge-deployable (Pi / Loihi class)
- **Real numbers** — run `python benchmark/run_benchmark.py` for judges who want data

### 5. Close (20 sec)
> "NeuroStream proves neuromorphic vision isn't theory — it's a working pipeline that processes **what changes, when it changes**."

---

## Judge Q&A — short answers

| Question | Answer |
|----------|--------|
| vs normal CNN? | We skip unchanged pixels; MAC count drops ~95% |
| Real event camera? | Same math; swap Stage 1 input for DVS hardware |
| STDP without labels? | Unsupervised — weights strengthen on causal spike pairs |
| Power claim? | MAC savings proxy; sparse ops ≈ idle MAC units on silicon |

---

## Tips
- Press **s** to toggle spikes if panel is too busy
- If laggy: close other apps; synthetic mode is already tuned
- Backup: run headless benchmark and read sparsity from terminal
