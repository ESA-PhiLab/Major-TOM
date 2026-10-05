"""GIF 3: why Major TOM window sizes come in multiples of 12 pixels.

Three strips show the same 480 m of ground as Sentinel-2 pixels at 10, 20 and 60 m. A window
(indigo) cuts whole pixels (mint) or splits pixels (yellow, would need resampling).
  Scene A: window grows from a corner on the 60 m grid, 1 px (10 m) at a time -> clean at ÷6.
  Scene B: window grows around a centre on the 60 m grid, 2 px at a time -> clean at ÷12.
  Scene C: real sizes 1044-1152 with ÷12 and ÷16 (unpadded GeoTIFF tiles) badges.
Loop: rest (empty strips) -> A -> B -> C -> rest (seam inside the rest).

Usage (from docs/animations):
  srun --cpus-per-task=4 --mem=16G python gif3_multiples_of_12.py [--review DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from style import (INDIGO, INK, INK_SOFT, MINT, PAPER, YELLOW, ease, mix, new_figure, phase, render_loop,
                   text, use_brand_fonts)

GROUND_M = 480                                    # ground shown in the strips (metres)
STRIPS = [  # pixel size (m), bottom of strip in axes units, label, bands
    (10, 3.0, "10 m bands", "B02  B03  B04  B08"),
    (20, 1.5, "20 m bands", "B05  B06  B07  B8A  B11  B12"),
    (60, 0.0, "60 m bands", "B01  B09  B10"),
]
AXES = [0.22, 0.36, 0.66, 0.44]                   # strips area in figure fractions
YLIM = (-0.4, 4.4)
CORNER_X0 = 120                                   # scene A: window starts at a 60 m grid line
CENTRE_X = 300                                    # scene B: window centred on a 60 m grid line
RULER = [1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152]


def fig_y(y: float) -> float:
    """Axes y (strip units) -> figure fraction, for labels placed next to the strips."""
    return AXES[1] + (y - YLIM[0]) / (YLIM[1] - YLIM[0]) * AXES[3]


def stepped(progress: float, n_steps: int, glide: float = 0.3) -> float:
    """0 -> n_steps in equal steps: each step glides (eased) in the first `glide` of its slot, then holds."""
    if progress >= 1:
        return float(n_steps)
    x = max(progress, 0.0) * n_steps
    k = int(x)
    return k + ease((x - k) / glide)


def linear(t: float, start: float, end: float) -> float:
    """Un-eased progress from t=start to t=end (the steps inside are eased instead)."""
    return float(np.clip((t - start) / (end - start), 0, 1))


def pixel_states(pixel_m: int, x0: float, x1: float) -> list[str]:
    """Per pixel of one strip: 'in' (whole pixel inside the window), 'cut' (split by an edge) or 'out'."""
    states = []
    for k in range(GROUND_M // pixel_m):
        p0, p1 = k * pixel_m, (k + 1) * pixel_m
        if x1 - x0 > 1e-6 and x0 <= p0 + 1e-6 and p1 <= x1 + 1e-6:
            states.append("in")
        elif x1 - x0 > 1e-6 and p0 < x1 - 1e-6 and p1 > x0 + 1e-6:
            states.append("cut")
        else:
            states.append("out")
    return states


def draw_strips(fig, x0: float, x1: float, level: float) -> list[bool]:
    """Strips with pixel states for window [x0, x1] (metres); returns per strip whether it is clean."""
    ax = fig.add_axes(AXES)
    ax.set_xlim(-4, GROUND_M + 4)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    clean = []
    for pixel_m, y0, _, _ in STRIPS:
        states = pixel_states(pixel_m, x0, x1)
        for k, state in enumerate(states):
            color = {"in": mix(PAPER, MINT, level), "cut": mix(PAPER, YELLOW, level), "out": PAPER}[state]
            ax.add_patch(Rectangle((k * pixel_m, y0), pixel_m, 1, facecolor=color, edgecolor=INK, lw=0.8))
        clean.append("cut" not in states)
    if level > 0.01 and x1 - x0 > 0.5:
        ax.add_patch(Rectangle((x0, -0.25), x1 - x0, 4.5, fill=False, edgecolor=mix(PAPER, INDIGO, level),
                               lw=3, zorder=5))
    return clean


def draw_badges(fig, clean: list[bool], level: float) -> None:
    """✓ / ✗ next to each strip."""
    for (_, y0, _, _), ok in zip(STRIPS, clean):
        text(fig, 0.905, fig_y(y0 + 0.5), "✓" if ok else "✗", size=22, weight=500, va="center", level=level,
             bbox=dict(boxstyle="circle,pad=0.25", facecolor=mix(PAPER, MINT if ok else YELLOW, level),
                       edgecolor="none"))


def draw_ruler(fig, t: float) -> None:
    """Scene C: real window sizes with their ÷12 and ÷16 badges, appearing one by one."""
    for i, size in enumerate(RULER):
        level = phase(t, 0.76 + 0.006 * i, 0.79 + 0.006 * i) - phase(t, 0.94, 0.97)
        x = 0.1 + i * 0.8 / (len(RULER) - 1)
        if size == 1056:
            text(fig, x, 0.305, "default", size=12, weight=500, color=INDIGO, ha="center", level=level)
        if size == 1068:
            text(fig, x, 0.305, "Core", size=12, weight=400, color=INK_SOFT, ha="center", level=level)
        frame = dict(boxstyle="round,pad=0.3", facecolor=PAPER, edgecolor=mix(PAPER, INDIGO, level), lw=2) \
            if size == 1056 else None
        text(fig, x, 0.255, str(size), size=19, weight=500, ha="center", level=level, bbox=frame)
        for y, divisor in ((0.195, 12), (0.145, 16)):
            ok = size % divisor == 0
            text(fig, x, y, f"÷{divisor} {'✓' if ok else '✗'}", size=13, weight=400, ha="center", level=level,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=mix(PAPER, MINT if ok else YELLOW, level),
                           edgecolor="none"))


def draw(t: float):
    """Frame at loop time t in [0, 1)."""
    # scene A: corner-anchored, 1 px steps;  scene B: centred, 2 px steps
    a_on = phase(t, 0.04, 0.06) - phase(t, 0.34, 0.38)
    b_on = phase(t, 0.40, 0.42) - phase(t, 0.70, 0.74)
    if t < 0.39:
        n_px, level, mode = stepped(linear(t, 0.06, 0.30), 12), a_on, "from a corner on the 60 m grid"
        x0, x1 = CORNER_X0, CORNER_X0 + 10 * n_px
    else:
        n_px, level, mode = 2 * stepped(linear(t, 0.42, 0.66), 12), b_on, "around a centre on the 60 m grid"
        x0, x1 = CENTRE_X - 5 * n_px, CENTRE_X + 5 * n_px

    fig = new_figure()
    clean = draw_strips(fig, x0, x1, level)
    for pixel_m, y0, label, bands in STRIPS:
        text(fig, 0.205, fig_y(y0 + 0.62), label, size=15, weight=400, ha="right", va="center")
        text(fig, 0.205, fig_y(y0 + 0.28), bands, size=10, color=INK_SOFT, ha="right", va="center")
    draw_badges(fig, clean, level if n_px > 0.05 else 0.0)
    shown = int(round(n_px))
    text(fig, 0.5, 0.905, f"Window: {shown} px = {10 * shown} m", size=26, weight=500, ha="center", level=level)
    text(fig, 0.5, 0.85, mode, size=15, color=INK_SOFT, ha="center", level=level)
    if t >= 0.39:                                                     # scene B: mark the centre
        text(fig, AXES[0] + AXES[2] * (CENTRE_X + 4) / (GROUND_M + 8), fig_y(4.33), "▼", size=14,
             color=INDIGO, ha="center", va="center", level=b_on)
    draw_ruler(fig, t)

    captions = [
        (1 - phase(t, 0.01, 0.04) + phase(t, 0.96, 0.99), "Sentinel-2 records its bands at 10, 20 and 60 m."),
        (phase(t, 0.06, 0.09) - phase(t, 0.35, 0.38),
         "From a corner on the 60 m grid: whole pixels in every band only when the size divides by 6."),
        (phase(t, 0.42, 0.45) - phase(t, 0.71, 0.74),
         "Around a centre on the 60 m grid: whole pixels only when the size divides by 12."),
        (phase(t, 0.77, 0.80) - phase(t, 0.93, 0.96),
         "Sizes that also divide by 16 tile into GeoTIFF blocks without padding: multiples of 48."),
    ]
    for lvl, caption in captions:
        text(fig, 0.5, 0.055, caption, size=17, weight=400, ha="center", level=lvl)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review", type=Path, help="also save a contact sheet of a few frames to this folder")
    args = p.parse_args()
    use_brand_fonts()
    render_loop(draw, seconds=16, name="gif3_multiples_of_12", review_dir=args.review,
                review_ts=(0.02, 0.17, 0.33, 0.55, 0.68, 0.88))


if __name__ == "__main__":
    main()
