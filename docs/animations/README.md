> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** GIF 1 reviewed (Miko, 2026-10-04); GIFs 0, 2–5 and the two slides unreviewed

# Tutorial animations

Looping GIFs and one static slide for the Major TOM tutorial, in the Asterisk Labs style (asterisk.coop/brand). Output in `out/`.

| GIF | Shows | Loop |
|---|---|---|
| `gif0_workflow.gif` | Overview: grid → footprint → archive query → read + cloud mask → rumi + GeoZL → TACO. The all-lit hold doubles as a static slide. | 12 s |
| `gif1_grid.gif` | How grid points are defined: rows in latitude, points per row, cells. Compared with a plain lat/lon grid. | 12 s |
| `gif2_anchoring.gif` | Same cell, same 1056 px window, two anchors (bottom-left grid point vs cell centre) as the cell rotates against UTM. | 14 s |
| `gif3_multiples_of_12.gif` | Why sizes are multiples of 12 px: whole 10/20/60 m pixels from a corner (÷6) and around a centre (÷12); ÷16 for GeoTIFF tiles. | 16 s |
| `gif4_tradeoff.gif` | Omission vs duplication as the window grows from 960 to 1152 px, on two patches and for all Core cells. | 18 s |
| `gif5a_same_product.gif` | Race A: the product Major TOM holds (cell 451U_946L, Snowbird, 2023-04-15) from five archives at measured speed, with requests and MB. | 14 s |
| `gif5b_newest_scene.gif` | Race B: the newest clear scene (last 60 days) from each archive; Major TOM returns its fixed 2023 sample. | 21 s |
| `archives_matrix.png` | Static slide: what each archive offers and asks for, plus the measured Snowbird results. | — |
| `poles_1056.png` | Static slide: 10 km cells and 1056 px windows at both poles, with caveats. | — |

## Files

```
style.py      colours, League Spartan, timing helpers, loop rendering + GIF writing + checks
geometry.py   official grid (MajorTOM/grid.py), 10 km cells, bottom-left and centroid windows
gifN_*.py     one script per GIF: draw(t) for t in [0, 1) -> style.render_loop
fonts/        League Spartan (SIL Open Font License, see fonts/OFL.txt)
tradeoff_curve.csv   GIF 4 curve, exported from the research table (see below)
acquire_snowbird.py  times the Snowbird sample per archive -> data/acquisition_runs.jsonl, data/chip_*.png
fig_archives.py      archive comparison slide (facts checked 2026-10-05)
fig_poles.py         pole slide (UPS windows, coverage counts)
```

## Regenerate

Run from this folder, through Slurm (CPU only):

```
srun --cpus-per-task=4 --mem=16G python geometry.py            # self-test against grid.py
srun --cpus-per-task=4 --mem=16G python gif1_grid.py --review /tmp/review
```

`--review DIR` also writes a contact sheet of six frames. Each render prints a seam check (change from the last frame back to the first) and a decode check (every frame of the written GIF compared with what was drawn).

`tradeoff_curve.csv` comes from `v2_misses_cells.parquet`, the per-cell table written by `docs/evidence/v2_misses.py` (not committed; large). To refresh it:

```
python gif4_tradeoff.py --export-curve PATH/v2_misses_cells.parquet
```

## Loop rules

- `draw(t)` is periodic over `t` in [0, 1); every animation rests in the same state at both ends.
- Frames at `t = i/n`, so the picture at `t = 1` is not drawn twice.
- One shared palette, no dithering: no colour flicker.
- Each frame stores only the pixels that changed (the rest is transparent), so files stay under 3 MB.

Environment: `miko-torch` (matplotlib 3.7, Pillow 9.4, pyproj, shapely 2, rasterio, geopandas 0.13 for the Natural Earth outlines in GIF 1).
