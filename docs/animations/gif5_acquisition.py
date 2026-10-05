"""GIF 5: getting one sample from five archives, played back at measured speed.

Cell 451U_946L (Snowbird, Utah; workshop venue), Sentinel-2 L2A, 1056 px window. Two races, from
acquire_snowbird.py (data/acquisition_runs.jsonl, successful runs only, medians per step):
  same    the product Major TOM holds (2023-04-15) from every archive  -> out/gif5a_same_product.gif
  recent  the newest clear scene in the last 60 days                  -> out/gif5b_newest_scene.gif
Each lane shows search, access, read and mask at real speed, the image developing while it reads,
then seconds, HTTP requests and MB fetched. Lanes without a working login show a lock.
Loop: rest (empty lanes, clock 0) -> race -> hold results -> fade -> rest.

Usage (from docs/animations):
  srun --cpus-per-task=4 --mem=16G python gif5_acquisition.py --race same [--review DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from style import (HERE, INDIGO, INDIGO_TINT, INK, INK_FAINT, INK_SOFT, MINT, PAPER, YELLOW, mix, new_figure,
                   render_loop, text, use_brand_fonts)

DATA = HERE / "data"
SPEED = 1.0                       # clock seconds per playback second (1.0 = real time, 0.5 = 2x slow motion)
T_MAX = 16.0                      # seconds shown on the timeline
REST, HOLD, FADE = 1.0, 4.0, 1.0
BAR = (0.25, 0.70)                # timeline x-range in figure fractions
TITLES = {"same": ("Same product, five archives", "15 April 2023, the product Major TOM Core holds for this cell"),
          "recent": ("Newest clear scene, five archives", "latest scene under 10% cloud in the last 60 days")}
LANES = [  # key, name, how it reads
    ("major-tom", "Major TOM Core", "Hugging Face · lookup + one row group"),
    ("planetary-computer", "Planetary Computer", "STAC · signed COG window reads"),
    ("earth-search", "AWS Earth Search", "STAC · COG window reads (us-west-2)"),
    ("cdse", "Copernicus Data Space", "STAC · JP2 window reads over S3"),
    ("gee", "Google Earth Engine", "computePixels on the exact grid"),
]
LANE_Y = [0.765, 0.625, 0.485, 0.345, 0.205]
STEP_STYLE = {"search": ("search", INDIGO_TINT), "access": ("access", YELLOW), "read": ("read 4 bands", INDIGO),
              "cloud mask": ("mask", MINT)}


def load(race: str) -> tuple[dict, dict]:
    """Per lane: median steps, bytes, requests, acquisition date; and per lane the errors seen."""
    ok, errors = defaultdict(list), defaultdict(list)
    for line in (DATA / "acquisition_runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["race"] == race:
            (errors if "error" in r else ok)[r["lane"]].append(r)
    lanes = {}
    for lane, runs in ok.items():
        names = [name for name, _ in runs[0]["steps"]]
        lanes[lane] = {"steps": [(n, statistics.median(dict(r["steps"])[n] for r in runs)) for n in names],
                       "bytes": runs[0]["bytes"], "requests": runs[0]["requests"], "acquired": runs[0]["acquired"],
                       "attempts": len(runs) + len(errors[lane])}
    return lanes, errors


def x_of(seconds: float) -> float:
    return BAR[0] + (BAR[1] - BAR[0]) * min(seconds, T_MAX) / T_MAX


def phase_in(seconds_since: float) -> float:
    """0 -> 1 over 0.3 s after an event."""
    return float(np.clip(seconds_since / 0.3, 0, 1))


def lock_text(lane: str, errors: list[dict]) -> str:
    if lane == "gee":
        return "Earth Engine login not set up on this machine yet"
    return f"failed: {errors[0]['error'][:60]}" if errors else "not measured"


def draw_lane(ax, fig, key: str, y: float, lane: dict | None, errors: list[dict], clock: float,
              chip: np.ndarray | None, level: float, race: str) -> None:
    """Timeline bar, developing image and result for one lane at `clock` seconds."""
    h = 0.05
    ax.add_patch(Rectangle((BAR[0], y - h / 2), BAR[1] - BAR[0], h, facecolor=mix(PAPER, INK_FAINT, 0.5),
                           edgecolor="none"))
    steps = lane["steps"] if lane else []
    start, read_progress = 0.0, 0.0
    for name, seconds in steps:
        shown = min(max(clock - start, 0.0), seconds)
        if shown > 0:
            label, color = STEP_STYLE[name]
            ax.add_patch(Rectangle((x_of(start), y - h / 2), x_of(start + shown) - x_of(start), h,
                                   facecolor=mix(PAPER, color, level), edgecolor="none"))
            if x_of(start + shown) - x_of(start) > 0.008 * len(label) + 0.01:       # label fits
                text(fig, x_of(start) + 0.005, y, label, size=11, weight=400, va="center", level=level,
                     color=PAPER if color == INDIGO else INK)
        if name == "read":
            read_progress = shown / seconds if seconds > 0 else 1.0
        start += seconds

    if lane is None and clock > 0:                                    # no successful run: lock box
        lock_level = level * min(1.0, clock / 0.4)
        ax.add_patch(FancyBboxPatch((BAR[0] + 0.004, y - h / 2), BAR[1] - BAR[0] - 0.004, h,
                                    boxstyle="round,pad=0,rounding_size=0.008", facecolor=PAPER,
                                    edgecolor=mix(PAPER, INK_SOFT, lock_level), lw=1.2, ls=(0, (3, 2))))
        text(fig, BAR[0] + 0.012, y, lock_text(key, errors), size=11, weight=400, va="center", color=INK_SOFT,
             level=lock_level)

    size = 0.105                                                      # thumbnail (figure fraction of height)
    w, x0, y0 = size * 720 / 1280, 0.725, y - size / 2
    ax.add_patch(Rectangle((x0, y0), w, size, facecolor=PAPER, edgecolor=INK_FAINT, lw=1))
    if chip is not None and read_progress > 0:
        rows = max(1, int(chip.shape[0] * read_progress))
        part = np.full_like(chip, 255)
        part[:rows] = chip[:rows]
        part = (255 - (255 - part.astype(np.float32)) * level).astype(np.uint8)
        ax.imshow(part, extent=(x0, x0 + w, y0, y0 + size), aspect="auto", zorder=3)

    if lane and clock >= start:                                       # finished: result + date
        done = level * phase_in(clock - start)
        text(fig, 0.80, y + 0.022, f"✓ {start:.1f} s", size=16, weight=600, color=INDIGO, va="center", level=done)
        req = f"{lane['requests']} requests · " if lane["requests"] else ""
        text(fig, 0.80, y - 0.008, f"{req}{lane['bytes'] / 1e6:.1f} MB", size=11, color=INK_SOFT, va="center",
             level=done)
        old = race == "recent" and key == "major-tom"
        label = f"{dt.date.fromisoformat(lane['acquired']):%d %b %Y}" + (" · fixed sample" if old else "")
        text(fig, 0.80, y - 0.036, label, size=11, weight=500 if old else 300, va="center", level=done,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=mix(PAPER, YELLOW, done), edgecolor="none") if old else None)
    if lane and errors:
        text(fig, 0.03, y - 0.046, f"search rate-limited (HTTP 429) in {len(errors)} of {lane['attempts']} attempts",
             size=10, color=INK_SOFT, va="center")


def draw(t: float, race: str, lanes: dict, errors: dict, chips: dict, loop: float, race_s: float):
    """Frame at loop time t in [0, 1)."""
    s = t * loop
    end_fade = REST + race_s + HOLD + FADE
    clock = 0.0 if s >= end_fade else float(np.clip((s - REST) * SPEED, 0, T_MAX))
    level = 0.0 if s >= end_fade else 1.0 - float(np.clip((s - REST - race_s - HOLD) / FADE, 0, 1))

    fig = new_figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title, subtitle = TITLES[race]
    text(fig, 0.03, 0.935, title, size=24, weight=500, va="center")
    text(fig, 0.03, 0.885, f"Cell 451U_946L (Snowbird, Utah)  ·  Sentinel-2 L2A  ·  1056 px window  ·  {subtitle}",
         size=13, color=INK_SOFT, va="center")
    text(fig, 0.965, 0.935, f"{clock:4.1f} s", size=28, weight=600, color=INDIGO, ha="right", va="center")
    text(fig, 0.965, 0.885, "real time" if SPEED == 1 else f"{1 / SPEED:.1f}× slow motion", size=12,
         color=INK_SOFT, ha="right", va="center")
    for sec in range(0, int(T_MAX) + 1, 4):
        ax.plot([x_of(sec)] * 2, [0.14, 0.835], color=INK_FAINT, lw=0.6, zorder=0)
        text(fig, x_of(sec), 0.845, f"{sec} s", size=10, color=INK_SOFT, ha="center")

    for (key, name, how), y in zip(LANES, LANE_Y):
        text(fig, 0.03, y + 0.016, name, size=16, weight=500, va="center")
        text(fig, 0.03, y - 0.018, how, size=11, color=INK_SOFT, va="center")
        draw_lane(ax, fig, key, y, lanes.get(key), errors.get(key, []), clock, chips.get(key), level, race)

    for i, (label, color) in enumerate(STEP_STYLE.values()):
        x = BAR[0] + i * 0.12
        ax.add_patch(Rectangle((x, 0.1), 0.018, 0.022, facecolor=color, edgecolor="none"))
        text(fig, x + 0.024, 0.111, label, size=12, weight=400, va="center")
    text(fig, 0.5, 0.04, "Medians of 3 runs from our server, 5 Oct 2026. Bands read concurrently over HTTP/2, "
                         "exact window only. Times depend on where you are.", size=12, color=INK_SOFT, ha="center")
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--race", choices=["same", "recent"], default="same")
    p.add_argument("--review", type=Path, help="also save a contact sheet of a few frames to this folder")
    args = p.parse_args()
    use_brand_fonts()
    lanes, errors = load(args.race)
    chips = {k: np.asarray(Image.open(DATA / f"chip_{args.race}_{k}.png").convert("RGB")).astype(np.float32)
             for k, *_ in LANES if (DATA / f"chip_{args.race}_{k}.png").exists()}
    lo, hi = np.percentile(np.stack(list(chips.values())), [2, 98])     # one stretch for the whole race
    chips = {k: (np.clip((c - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8) for k, c in chips.items()}
    race_s = min(T_MAX, max(sum(s for _, s in lane["steps"]) for lane in lanes.values()) + 0.8) / SPEED
    loop = REST + race_s + HOLD + FADE + 0.5
    name = {"same": "gif5a_same_product", "recent": "gif5b_newest_scene"}[args.race]
    render_loop(lambda t: draw(t, args.race, lanes, errors, chips, loop, race_s), seconds=loop, name=name,
                review_dir=args.review, review_ts=(0.02, 0.12, 0.25, 0.45, 0.8, 0.97))


if __name__ == "__main__":
    main()
