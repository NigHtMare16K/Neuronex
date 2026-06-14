#!/usr/bin/env python3
"""Headless benchmark — reports sparsity and MAC savings over N seconds."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import run


def main() -> int:
    p = argparse.ArgumentParser(description="NeuroStream headless benchmark")
    p.add_argument("--duration", type=float, default=10, help="Seconds to run")
    p.add_argument("--source", default="synthetic")
    ns = p.parse_args()

    class Args:
        source = ns.source
        headless = True
        duration = ns.duration

    return run(Args())


if __name__ == "__main__":
    raise SystemExit(main())
