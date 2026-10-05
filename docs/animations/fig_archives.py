"""Static slide: where to get Sentinel data, compared (out/archives_matrix.png).

Facts checked 2026-10-05 against each archive's STAC /collections and documentation (sources in
docs/reports/2026-10-05-acquisition-gifs.md). Bottom row: GIF 5's measured Snowbird times.
Cell tone: mint = easy, yellow = friction, white = neutral.

Usage (from docs/animations):  python fig_archives.py
"""
from __future__ import annotations

from matplotlib.patches import Rectangle

from acquire_snowbird import NEEDS_LOGIN
from style import (INDIGO, INK, INK_FAINT, INK_SOFT, MINT, OUT, PAPER, YELLOW, lock_after, mix, new_figure, text,
                   use_brand_fonts)

COLUMNS = ["AWS Earth Search", "Copernicus Data Space", "Google Earth Engine", "Planetary Computer",
           "Major TOM Core"]
COLUMN_KEYS = ["earth-search", "cdse", "gee", "planetary-computer", "major-tom"]
G, Y_, N = "good", "friction", "neutral"
ROWS = [  # label, one (text, tone) per column
    ("Sentinel-2 L1C", [("requester-pays JP2", Y_), ("✓ SAFE / JP2", G), ("✓", G), ("✗ not offered", Y_),
                        ("✓ Core-S2L1C", G)]),
    ("Sentinel-2 L2A", [("✓ free COG", G), ("✓ SAFE / JP2", G), ("✓ from 2017", G), ("✓ COG", G),
                        ("✓ Core-S2L2A", G)]),
    ("Sentinel-1", [("GRD, no RTC", N), ("GRD; gamma0 on the fly", N), ("GRD, terrain-corrected", N),
                    ("✓ GRD + RTC", G), ("✓ Core-S1RTC", G)]),
    ("Login", [("none", G), ("account + S3 keys", Y_), ("Cloud project + registration", Y_), ("none (free token)", G),
               ("none", G)]),
    ("Read just a window", [("✓ COG range reads", G), ("JP2 over S3, ~220 requests", Y_), ("computePixels ≤ 48 MB", N),
                            ("✓ COG range reads", G), ("one row group per sample", G)]),
    ("Limits", [("none documented", N), ("4 × 20 MB/s, 12 TB/month", Y_), ("40 parallel requests", N),
                ("token lasts ~45 min", N), ("—", N)]),
    ("L2A +1000 offset", [("kept, declared", G), ("kept, declared", G), ("removed (harmonised)", N),
                          ("kept, not declared", Y_), ("kept (as ESA L2A)", N)]),
    ("Same product, measured", [("6.7 s", Y_), ("4.8 s", N), ("4.7 s", N), ("1.8 s", G),
                                ("4.0 s", N)]),
    ("Newest clear scene", [("1 Oct 2026", G), ("1 Oct 2026 · 429s", N), ("1 Oct 2026", G), ("1 Oct 2026", G),
                            ("15 Apr 2023, fixed", Y_)]),
]
TONE = {G: mix(PAPER, MINT, 0.85), Y_: mix(PAPER, YELLOW, 0.45), N: PAPER}


def main() -> None:
    use_brand_fonts()
    fig = new_figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(fig, 0.03, 0.93, "Where to get Sentinel data", size=26, weight=500, va="center")
    text(fig, 0.03, 0.88, "Same satellites, different archives: what each one asks of you", size=14,
         color=INK_SOFT, va="center")
    x0, label_w, col_w = 0.03, 0.17, 0.152
    top, row_h = 0.80, 0.076
    for j, name in enumerate(COLUMNS):
        x = x0 + label_w + j * col_w + (0.012 if j == 4 else 0)        # small gap before Major TOM
        header = text(fig, x + col_w / 2 - (0.008 if COLUMN_KEYS[j] in NEEDS_LOGIN else 0), top, name, size=13,
                      weight=500, ha="center", va="center", color=INDIGO if j == 4 else INK)
        if COLUMN_KEYS[j] in NEEDS_LOGIN:
            lock_after(fig, header, size=12, gap=0.004)
    for i, (label, cells) in enumerate(ROWS):
        y = top - (i + 1) * row_h
        text(fig, x0, y, label, size=13, weight=400, va="center")
        for j, (value, tone) in enumerate(cells):
            x = x0 + label_w + j * col_w + (0.012 if j == 4 else 0)
            ax.add_patch(Rectangle((x + 0.003, y - row_h / 2 + 0.006), col_w - 0.006, row_h - 0.012,
                                   facecolor=TONE[tone], edgecolor=INK_FAINT if tone == N else "none", lw=0.6))
            text(fig, x + col_w / 2, y, value, size=11.5, weight=500 if i == len(ROWS) - 1 else 300,
                 ha="center", va="center")
    text(fig, 0.03, 0.035, "Checked 5 Oct 2026 against each archive's STAC catalogue and documentation. "
                           "Times from a server in the UK: sums of per-step medians over 3 runs for cell 451U_946L (GIF 5); 429s = CDSE search rate-limited 2 of 4 tries.",
         size=10.5, color=INK_SOFT, va="center")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "archives_matrix.png", dpi=200, facecolor=PAPER)
    print("wrote out/archives_matrix.png")


if __name__ == "__main__":
    main()
