"""Shared look (Asterisk Labs brand) and seamless-loop GIF writing for the tutorial animations.

Brand (asterisk.coop/brand): Ink / Paper / Indigo / Mint / Yellow, typeface League Spartan.

Loop rules, followed by every GIF:
  - draw(t) is periodic in t over [0, 1): whatever moves ends where it started;
  - frames are drawn at t = i/n, so t = 1 (the same picture as t = 0) is not drawn twice;
  - one shared palette for all frames, no dithering, so colours never flicker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import font_manager
from PIL import Image, ImageSequence

HERE = Path(__file__).resolve().parent
FONTS = HERE / "fonts"
OUT = HERE / "out"

W, H, DPI, FPS = 1280, 720, 100, 20


def mix(c1: str, c2: str, f: float) -> str:
    """Hex colour a fraction `f` of the way from `c1` to `c2` (flat colour, no transparency)."""
    a, b = np.array(mcolors.to_rgb(c1)), np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1 - f) * a + f * b)


# Brand palette
INK, PAPER = "#000000", "#ffffff"
INDIGO, MINT, YELLOW = "#492ae8", "#c4ffc2", "#f7cc09"
# Tints (mixed with paper) for secondary elements
INK_SOFT = mix(INK, PAPER, 0.55)      # secondary text, grid rows
INK_FAINT = mix(INK, PAPER, 0.82)     # coastlines, background lines
INDIGO_TINT = mix(INDIGO, PAPER, 0.6)
BRAND = [INK, PAPER, INDIGO, MINT, YELLOW, INK_SOFT, INK_FAINT, INDIGO_TINT]
CLEAR = 255     # palette slot reserved for "unchanged since the previous frame" (transparent)


# ---------- look ----------

def use_brand_fonts() -> None:
    """Register League Spartan from fonts/ and make it the default; DejaVu Sans fills missing glyphs."""
    for ttf in sorted(FONTS.glob("LeagueSpartan-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    # fail loudly instead of silently falling back to another font
    font_manager.findfont(font_manager.FontProperties(family="League Spartan"), fallback_to_default=False)
    plt.rcParams.update({"font.family": ["League Spartan", "DejaVu Sans"], "font.weight": 300,
                         "text.color": INK})


def new_figure() -> plt.Figure:
    """Blank 1280x720 canvas on paper white."""
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)


def text(fig: plt.Figure, x: float, y: float, s: str, size: float = 16, weight: int = 300,
         color: str = INK, level: float = 1.0, **kw) -> None:
    """Figure text that fades in by mixing from paper (`level` 0..1); skipped while invisible.

    Weights follow the brand: 300 body, 400 labels, 500 headings, 600 strong emphasis.
    """
    if level > 0.01:
        return fig.text(x, y, s, fontsize=size, fontweight=weight, color=mix(PAPER, color, level), **kw)
    return None


LOCK = "\U0001F512"     # padlock; drawn from Noto Sans Symbols2 (monochrome outline font)


def lock_after(fig: plt.Figure, artist, size: float = 14, gap: float = 0.006, color: str = INK) -> None:
    """Padlock right after a drawn text (marks archives that need an account)."""
    if artist is None:
        return
    box = artist.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(box.x1 + gap, (box.y0 + box.y1) / 2, LOCK, fontsize=size, color=color,
             fontfamily="Noto Sans Symbols2", va="center")


# ---------- timing ----------

def ease(x: float) -> float:
    """Smoothstep: rises 0 -> 1 with zero speed at both ends."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def phase(t: float, start: float, end: float) -> float:
    """Eased progress of a phase running from t=start to t=end: 0 before, 1 after."""
    return ease((t - start) / (end - start))


# ---------- writing the loop ----------

def to_image(fig: plt.Figure) -> Image.Image:
    """Rasterise a figure to RGB and close it."""
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def shared_palette(samples: list[Image.Image]) -> Image.Image:
    """One palette for every frame: brand colours exact, the rest fitted to sample frames.

    Slot CLEAR is set to magenta, far from every colour used, so no pixel is ever mapped to it.
    """
    mosaic = Image.new("RGB", (W, H * len(samples)))
    for i, img in enumerate(samples):
        mosaic.paste(img, (0, i * H))
    k = CLEAR - len(BRAND)
    fitted = mosaic.quantize(colors=k, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE).getpalette()[: 3 * k]
    exact = [round(v * 255) for c in BRAND for v in mcolors.to_rgb(c)]
    palette = Image.new("P", (1, 1))
    palette.putpalette((exact + fitted + [0] * 765)[:765] + [255, 0, 255])
    return palette


def to_palette(img: Image.Image, palette: Image.Image) -> Image.Image:
    """Map an RGB frame to `palette` by exact nearest colour (Pillow's own lookup is ~6-bit and
    turned pure white into #fcfcfc). Brand colours come first in the palette, so they win ties."""
    rgb_palette = np.array(palette.getpalette()[: 3 * CLEAR], dtype=np.int64).reshape(-1, 3)   # CLEAR excluded
    flat = np.asarray(img).reshape(-1, 3).astype(np.int64)
    colours, inverse = np.unique(flat @ np.array([65536, 256, 1]), return_inverse=True)  # each colour once
    rgb = np.stack([colours >> 16, (colours >> 8) & 255, colours & 255], axis=1)
    nearest = np.concatenate([((part[:, None, :] - rgb_palette[None]) ** 2).sum(-1).argmin(1)   # in chunks:
                              for part in np.array_split(rgb, max(1, len(rgb) // 8192))])       # photos have many colours
    out = Image.fromarray(nearest[inverse].reshape(img.height, img.width).astype(np.uint8), mode="P")
    out.putpalette(palette.getpalette())
    return out


def seam_report(frames: list[Image.Image]) -> str:
    """Pixel change from the last frame back to the first, against the change between neighbours."""
    def step(a: Image.Image, b: Image.Image) -> float:
        return float(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).mean())
    steps = [step(a, b) for a, b in zip(frames, frames[1:])]
    worst = int(np.argmax(steps))
    return (f"seam {step(frames[-1], frames[0]):.3f} | neighbouring frames: median {np.median(steps):.3f}, "
            f"max {steps[worst]:.3f} (t = {worst / len(frames):.3f} -> {(worst + 1) / len(frames):.3f})")


def contact_sheet(frames: list[Image.Image], ts: tuple[float, ...], path: Path, scale: float = 0.5) -> None:
    """Frames at loop times `ts`, three per row, as one PNG for review."""
    w, h, gap = int(W * scale), int(H * scale), 6
    picks = [frames[int(t * len(frames)) % len(frames)] for t in ts]
    n_rows = -(-len(picks) // 3)
    sheet = Image.new("RGB", (3 * w + 2 * gap, n_rows * h + (n_rows - 1) * gap), INK_FAINT)
    for i, img in enumerate(picks):
        sheet.paste(img.resize((w, h), Image.Resampling.LANCZOS), ((i % 3) * (w + gap), (i // 3) * (h + gap)))
    sheet.save(path)


def render_loop(draw: Callable[[float], plt.Figure], seconds: float, name: str, fps: int = FPS,
                review_dir: Path | None = None,
                review_ts: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)) -> Path:
    """Render draw(t) for t = i/n in [0, 1) and save out/<name>.gif, looping forever."""
    n = round(seconds * fps)
    frames = [to_image(draw(i / n)) for i in range(n)]
    print(seam_report(frames))
    if review_dir:
        review_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet(frames, review_ts, review_dir / f"{name}_sheet.png")
    palette = shared_palette(frames[:: max(1, n // 24)])
    frames = [to_palette(f, palette) for f in frames]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.gif"
    write_gif(frames, path, fps)
    print(f"{path.relative_to(HERE)}: {n} frames, {path.stat().st_size / 1e6:.1f} MB")
    print(verify_gif(path, frames, fps))
    return path


def write_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    """Save palette frames as a looping GIF that stores only the pixels each frame changes.

    Pixels equal to the previous frame are set to the transparent slot CLEAR; with disposal=1
    the viewer keeps the previous frame underneath, so the picture is unchanged.
    """
    deltas = [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        a = np.asarray(cur)
        delta = Image.fromarray(np.where(a == np.asarray(prev), CLEAR, a).astype(np.uint8), mode="P")
        delta.putpalette(cur.getpalette())
        deltas.append(delta)
    deltas[0].save(path, save_all=True, append_images=deltas[1:], duration=1000 // fps, loop=0,
                   transparency=CLEAR, disposal=1, optimize=False, background=1)   # slot 1 = PAPER


def verify_gif(path: Path, frames: list[Image.Image], fps: int) -> str:
    """Decode the written GIF and compare every frame a viewer would see with what was rendered."""
    shown = []
    with Image.open(path) as gif:
        for f in ImageSequence.Iterator(gif):                        # merged frames carry longer durations
            shown += [f.convert("RGB")] * round(f.info["duration"] * fps / 1000)
    worst = max(int(np.abs(np.asarray(a, np.int16) - np.asarray(b.convert("RGB"), np.int16)).max())
                for a, b in zip(shown, frames))
    corner = shown[0].getpixel((5, 5))
    return f"decoded check: {len(shown)} of {len(frames)} frames, max pixel error {worst}, corner {corner}"
