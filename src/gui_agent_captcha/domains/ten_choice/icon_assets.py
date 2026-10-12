from __future__ import annotations

import math
import random
from pathlib import Path

ICON_NAMES = [
    "alloy",
    "arrowhead",
    "astrolabe",
    "axial",
    "barb",
    "boulder",
    "branch",
    "canopy",
    "capsule",
    "cascade",
    "cipher",
    "circuit",
    "coil",
    "cusp",
    "delta",
    "drift",
    "echo",
    "facet",
    "fan",
    "filament",
    "finch",
    "fracture",
    "glide",
    "graviton",
    "gridline",
    "hook",
    "ion",
    "keystone",
    "lantern",
    "loop",
    "marrow",
    "mast",
    "nexus",
    "obelisk",
    "paddle",
    "pendant",
    "petal",
    "pike",
    "pulse",
    "radar",
    "reed",
    "relic",
    "rosette",
    "sail",
    "scale",
    "scythe",
    "shell",
    "shiver",
    "spoke",
    "surge",
    "talon",
    "tempo",
    "thread",
    "torch",
    "vector",
    "vein",
    "wedge",
    "whirl",
    "yoke",
    "zephyr",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ICON_DIR = REPOSITORY_ROOT / "apps/hover-reveal/public/icons"
OUTPUT_DIR = DEFAULT_ICON_DIR


def _blob_points(rng: random.Random, *, count: int = 7) -> str:
    points: list[str] = []
    for index in range(count):
        angle = (2 * math.pi * index) / count
        radius = rng.uniform(15.0, 24.0)
        x = 32 + math.cos(angle) * radius
        y = 32 + math.sin(angle) * radius
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _accent_path(rng: random.Random) -> str:
    start_x = rng.uniform(14.0, 24.0)
    start_y = rng.uniform(16.0, 26.0)
    ctrl1_x = rng.uniform(22.0, 30.0)
    ctrl1_y = rng.uniform(10.0, 22.0)
    ctrl2_x = rng.uniform(34.0, 42.0)
    ctrl2_y = rng.uniform(40.0, 50.0)
    end_x = rng.uniform(40.0, 50.0)
    end_y = rng.uniform(34.0, 46.0)
    return (
        f"M {start_x:.1f} {start_y:.1f} "
        f"C {ctrl1_x:.1f} {ctrl1_y:.1f}, "
        f"{ctrl2_x:.1f} {ctrl2_y:.1f}, "
        f"{end_x:.1f} {end_y:.1f}"
    )


def _secondary_path(rng: random.Random) -> str:
    left_x = rng.uniform(16.0, 22.0)
    left_y = rng.uniform(36.0, 46.0)
    mid_x = rng.uniform(28.0, 34.0)
    mid_y = rng.uniform(22.0, 34.0)
    right_x = rng.uniform(40.0, 48.0)
    right_y = rng.uniform(18.0, 30.0)
    return (
        f"M {left_x:.1f} {left_y:.1f} "
        f"L {mid_x:.1f} {mid_y:.1f} "
        f"L {right_x:.1f} {right_y:.1f}"
    )


def build_svg(name: str) -> str:
    rng = random.Random(name)
    blob_points = _blob_points(rng, count=rng.randint(6, 9))
    accent_path = _accent_path(rng)
    secondary_path = _secondary_path(rng)
    circle_x = rng.uniform(22.0, 42.0)
    circle_y = rng.uniform(20.0, 42.0)
    circle_r = rng.uniform(3.5, 6.5)
    stroke_width = rng.uniform(3.2, 4.6)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">\n'
        f'  <polygon points="{blob_points}" fill="#0f172a"/>\n'
        f'  <path d="{accent_path}" fill="none" stroke="#e2e8f0" '
        f'stroke-width="{stroke_width:.1f}" stroke-linecap="round"/>\n'
        f'  <path d="{secondary_path}" fill="none" stroke="#94a3b8" '
        'stroke-width="3.0" stroke-linecap="round" stroke-linejoin="round"/>\n'
        f'  <circle cx="{circle_x:.1f}" cy="{circle_y:.1f}" r="{circle_r:.1f}" fill="#e2e8f0"/>\n'
        "</svg>\n"
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in ICON_NAMES:
        path = OUTPUT_DIR / f"{name}.svg"
        path.write_text(build_svg(name), encoding="utf-8")
        written += 1
    print(f"Wrote {written} procedural icons to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
