"""
region_mapper.py
Maps spike clusters (neuron grid space) back to
frame pixel coordinates for overlay rendering.

Also provides zone labelling — divides the frame into
named spatial zones (TOP_LEFT, CENTER, etc.) to give
human-readable location context to each MotionRegion.
"""

import numpy as np
from detection.pattern_detector import MotionRegion
from config.settings import FRAME_WIDTH, FRAME_HEIGHT


# ── Zone grid (3×3) ────────────────────────────────────────────────
ZONES = [
    ["TOP_LEFT",    "TOP_CENTER",    "TOP_RIGHT"   ],
    ["MID_LEFT",    "CENTER",        "MID_RIGHT"   ],
    ["BOT_LEFT",    "BOT_CENTER",    "BOT_RIGHT"   ],
]


class RegionMapper:
    """
    Translates MotionRegion pixel positions to:
      - Named screen zone (e.g. "TOP_LEFT")
      - Normalised (0–1) coordinates
      - Bounding box for overlay
    """

    def __init__(self,
                 frame_w: int = FRAME_WIDTH,
                 frame_h: int = FRAME_HEIGHT):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.zone_w  = frame_w // 3
        self.zone_h  = frame_h // 3

    # ── Mapping ────────────────────────────────────────────────────
    def label_region(self, region: MotionRegion) -> str:
        """Return zone name for the region's centre pixel."""
        col = min(int(region.center_x // self.zone_w), 2)
        row = min(int(region.center_y // self.zone_h), 2)
        return ZONES[row][col]

    def to_normalised(self, region: MotionRegion) -> tuple[float, float]:
        """Return (nx, ny) where each value is in [0, 1]."""
        nx = region.center_x / self.frame_w
        ny = region.center_y / self.frame_h
        return (nx, ny)

    def enrich_regions(self, regions: list[MotionRegion]) -> list[dict]:
        """
        Return list of enriched region dicts ready for the visualiser.
        Each dict contains: bbox, zone, center, spike_count, label.
        """
        enriched = []
        for region in regions:
            enriched.append({
                "bbox"        : region.to_bbox(),
                "zone"        : self.label_region(region),
                "center"      : (int(region.center_x), int(region.center_y)),
                "spike_count" : region.spike_count,
                "label"       : region.label,
                "normalised"  : self.to_normalised(region),
            })
        return enriched

    def zone_grid(self) -> list[dict]:
        """
        Return bounding boxes for all 9 zones — useful for debug overlay.
        """
        zones = []
        for r in range(3):
            for c in range(3):
                zones.append({
                    "name": ZONES[r][c],
                    "bbox": (
                        c * self.zone_w, r * self.zone_h,
                        (c + 1) * self.zone_w, (r + 1) * self.zone_h
                    )
                })
        return zones
