> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-11-04 — Intro animation v2 (Major TOM Climate, Core-Weather)

Rebuilt `docs/animations/intro_climate_weather.py` after review of v1. Output: 66.0 s, 30 fps, H.264 yuv420p, in two
layouts from one code path.

- `/data/Major-TOM/animations/intro-v2/major-tom-climate-weather-v2-square.mp4`: 1080 x 1080, 7,119,587 bytes
- `/data/Major-TOM/animations/intro-v2/major-tom-climate-weather-v2-wide.mp4`: 1920 x 1080, 8,563,813 bytes
- Copies and 30 key frames (`intro2-<layout>-<scene>.png`, with `flag-fill`, `flag-centre`, `grid-zoom`, `normals-six` and `size`) in `screenshots/` (git-ignored)

Not committed. v1 output in `/data/Major-TOM/animations/intro/` is untouched; the v1 script is replaced.

## Scenes

| Time (s) | Scene | Data |
|---|---|---|
| 0–7.6 | Flag fills large and centred, cell by cell (observations, then the fill); moves into the logo; "Major TOM" fades in, then "climate"; both dataset names | `apps/climate-globe` loader |
| 7.6–15.2 | Globe centred on Loch Lomond zooms in to 21 rows of Major TOM cells; cells fill column by column, top to bottom, in the flag colours; 625U_30L picked out and labelled | `majortom/cells.py`, Natural Earth 1:10m |
| 15.2–19.2 | Card "Dataset 1 of 2: Major TOM Climate" (flag half lit) | — |
| 19.2–26.7 | ERA5 2 m temperature normals on one globe, which shrinks into six globes with five more normals, months in sync at 0.3 s each; "294 variables in 6 datasets" | `climatology_overviews/` of ERA5, IMERG, CAMS, ERA5-Land |
| 26.7–31.1 | ERA5 monthly 2 m temperature at Loch Lomond, 1940–2025, with 12-month mean | ERA5-Monthly `by_cell/band=16` (local) |
| 31.1–35.1 | Card "Dataset 2 of 2: Major TOM Core-Weather" (flag full) | — |
| 35.1–43.1 | Loch Lomond S2 L2A image, hourly 2 m temperature (ERA5, ERA5-Land) and total cloud cover (ERA5) over ±24 h | Core-S2L2A, Core-Weather `hourly/` |
| 43.1–50.1 | Sources by kind of data, an emoji per kind and per main quantity: reanalysis, satellite, rain gauges | READMEs, Noto Color Emoji |
| 50.1–57.2 | Code types out; CPC series draws over the measured read time; four counters (time, requests, bytes, months) | live read |
| 57.2–62.7 | Size and access: Climate 5.83 TB and Core-Weather 395.4 GB counting up, then the format lines | `du` |
| 62.7–66.0 | End card: logo, URLs, Earth Compress and Asterisk Labs, credit, Copernicus line | brand files |

## docs/animations/intro_climate_weather.py

Positions come from one table per layout (`:92`). Scenes read boxes from `L`, set per frame.

    LAYOUTS = {
        "square": {"size": (1080, 1080), "margin": 48, "logo": (90, 400, 900), ...
        "wide": {"size": (1920, 1080), "margin": 96, "logo": (360, 360, 1200), ...

The flag is the app's 20 loader polygons and colours, lit one per 70 ms in the app's order (`:73`).

Globe is an orthographic projection by hand (`:460`); cartopy is not in `miko-torch`. Each pixel maps to a grid row
from latitude and a grid column from distance along the parallel, then to the 4 x 4 block of the overviews.

    lat = np.arcsin(np.clip(z * np.sin(p0) + y * np.cos(p0), -1, 1))
    lon = l0 + np.arctan2(x, z * np.cos(p0) - y * np.sin(p0))
    ...
    col = np.floor(lon * g["r_km"] * np.cos(lat) / g["d_km"]).astype(int)

Grid scene (`:684`), replaced after review: the first version (turning globe, colour strips, a line every 32
columns) did not show the grid. The new one zooms from the globe into Loch Lomond. Cells come from
`majortom/cells.py` (`cell_from_name`, `cells_in`, `Cell.lon/.lat/.dlon/.dlat`), loaded as a single module
(`:358`) because `majortom/__init__.py` needs Python 3.10 and `miko-torch` has 3.8.

    table = cells_in((west, south, east, north))
    self.cells = [Cell(int(r), int(c)) for r, c in zip(table.row, table.col)]
    ...
    self.fill_band = np.array([c.col % 4 for c in self.cells])

21 rows (615U–635U), in each row the cells within 10.5 of 625U_30L's column widths: 469 cells. Fill order is by
column number, then north to south. Colour is column number modulo 4 in the flag's four colours, so each column's
shift between rows shows. At 30 columns west of the prime meridian the shift is about 0.7 km per row, plus a jump of
about half a cell at one row near the target (the row's column count changes).

Coastlines: Basemap's low-resolution world lines while the view is wide; Natural Earth 1:10m land and European lakes
(`shapes` step, `:158`, `python-last`, cartopy downloader) once the view is within 300 km. Basemap's installed
resolutions have no lakes near Loch Lomond; `basemap-data-hires` is not installed. Loch Lomond spans about one cell
north and 1.5 cells south of 625U_30L, matching 56.0–56.33 N against the cell's 56.14–56.23 N.

Sources scene (`:1148`): one emoji per kind of data and per main quantity, as additions to the text labels. Noto
Color Emoji is a bitmap font matplotlib cannot draw, so `emojis` (`:237`) renders each with PIL at 109 px (the font's
only size) onto one 140 x 128 canvas, so all share a baseline, and `_emoji` (`:1124`) places them as images scaled to
the text size. Variation selectors are left out: PIL here has no raqm.

    font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)
    ...
    ImageDraw.Draw(im).text((2, 2), char, font=font, embedded_color=True)

Emojis: 🧮 reanalysis, 🛰 satellite, 🌧 rain gauges; 🌡 temperature, 🌧 precipitation, 💨 wind, ❄ snow,
💧 soil water, 🌫 air quality (CAMS aerosols, PM2.5, trace gases; Sentinel-5P trace gases). The variable counts (201,
49) moved to each tile's last line: on the grid line they ran past the square tile's edge.

Access split: the read scene keeps the code, the series and four counters (2 x 2 under the code in the wide layout,
where a row ran into the chart). `scene_size` (`:1255`) is new: sizes count up over 1.4 s, then the two format lines.
Timing after review: normals 5.0 s (0.7 s intro, 0.3 s a month, cross-fade over the last 60 % of each month). Cards
4.0 s, text fully visible 3.0 s. Explanatory text fully visible at least 2.5 s before its scene fades: flag
subtitle (2.6 s; flag 6.0 s, wordmark earlier), grid captions (4.0 and 2.5 s; grid 7.6 s), the weather
no-precipitation line (2.6 s), the read counters (2.5 s; read 7.1 s), the size scene's format lines (2.6 s). To stay
under 65 s: series 6.0 s (drawn in 4.4 s), weather 8.0 s, sources 7.0 s, end card 4.3 s.

Flag fill (`:561`, `:605`), replacing the bar-by-bar loader fill after review (prototype 3b): each bar has its
own grid of 10 x 4 cells (8 x 4 in the outer columns), bilinear in the bar's corners so cells follow its slant; 720
cells. About 20 % are observations (each bar's share 3–35 %), which light first; the rest light by distance to the
nearest observation in their bar. Each cell fades in over 0.1 s in the bar's own colour, edged in that colour and
clipped to the bar, so nothing crosses a bar border and no seams show.

Timing fix: the first version gave fill cells a start of 0.8 s plus distance / speed, with the speed set by the
sparsest bar. Nothing started for 9 frames after the observations (0.83–1.07 s), then 294 cells started within 3
frames (up to 135 in one), because distances come in whole cell steps; peak mean colour change 2.97 per frame,
against about 0.2 before. Now the cells are ranked (observations, then distance plus up to 0.6 cell width of
jitter) and the k-th of 720 starts at 2.4 x inverse-smoothstep((k + 0.5) / 720) after 0.1 s: at most 15 cells start
per frame, no frame of the fill is static, peak change 0.40. Plot: screenshots/flag-proto-3b-timing.png. The
encode was not at fault (90 frames, 30 fps). The last cell is lit at 2.56 s; the move starts at 2.6 s and its first
frame equals the fill's last (difference 0 in both layouts).

Final fill timing: observations start at a sin² rate over 0.1–0.7 s, as their own step (up to 18 cells a frame);
the fill starts at 0.4 s at a rate x (1 - x)^0.9, slow then accelerating (up to 13 a frame), then tapering to 2.47
s. The rates overlap, so the count dips to 6 between the steps and never stops. Plot (final curve):
screenshots/flag-proto-3b-timing.png. The prototype had been blending to the main video's frame 48, which became
mid-fill once the fill moved into the main video; it now uses frame 78 (2.6 s, fill done, not yet moving), and its
end matches that frame exactly. A bar mid-fill shows a grey-tinted 1 px outline (its grey base under the clipped
cells, up to 70 levels); it goes when the bar completes, the last at 2.6 s, the frame before the move's first step.

Other changes after review: 🏭 for air quality (CAMS, Sentinel-5P) instead of 🌫; cloud cover scale inverted (more
cloud lighter, `#36435a` to `#eaeef4`, one hue); the size scene gains the Major TOM Climate icon beside its number
(docs/brand/mt-climate-icon.svg, through the logos step; no Core-Weather mark exists in the repo, so its block has
none) and three format lines with 📦, 🌐, 🐍.

Palette checks, now run (Node 20 in a new conda env `node20`, the dataviz validator):
- Chart slots `#2a78d6`, `#eb6834`, `#1baf7a`, all pairs: pass; CPC green warns on contrast (2.74:1); its lines carry
  direct labels.
- ERA5 blue with ERA5-Land grey `#52514e`: fails the chroma floor (grey reads as no hue); kept from the README
  figures, lines direct-labelled.
- Map ramps (`--ordinal`): all fail the light-end-versus-page check (it is for marks on the page; these fill globes
  with an outline). Snow, soil water, cloud (after the tint) and PM2.5 pass monotone lightness and single hue;
  precipitation (DRY_WET) is multi-hue and temperature (COLD_HOT) diverging, both by design from the README figures.

Flag scene (`:638`), reworked after review: the 20 bars fill large and centred (45 % of the canvas height) in the
app's order and timing, then move and scale into the logo (eased 0.8 s, size interpolated geometrically). "Major
TOM" fades in, and "climate" 0.4 s after it, from the one wordmark image split at x = 392 logo units ("Major TOM"
ends at 381, "climate" starts at 403.5). No wipe. At the end of the move the bars are pixel-identical to the final
hold (difference 0 in both layouts).

Normals scene (`:891`), reworked: one globe of ERA5 2 m temperature for 1.8 s, then it shrinks (eased 0.7 s) into
the first slot of a 3 x 2 grid and five more normals fade in, all on the same month (0.3 s a month, continuing past
December). `six` (`:863`) reads them from each dataset's `climatology_overviews/`:

| Globe | Column | Shown as | Scale |
|---|---|---|---|
| 2 m temperature, ERA5 | `era5_t2m_mean` | °C | diverging at 0 °C (COLD_HOT) |
| Precipitation, IMERG | `imerg_precip_mean` | mm/day (x 24) | DRY_WET, log 0.1–15 |
| Surface PM2.5, CAMS | `cams_pm2p5_mean` | µg m⁻³ (x 1e9) | HAZE, log 1–100 |
| Snow cover, ERA5-Land | `era5land_snowc_mean` | % | one hue blue, 0–100 |
| Soil water, ERA5-Land | `era5land_swvl1_mean` (top layer, 0–7 cm) | m³ m⁻³ | one hue teal, 0.02–0.5 |
| Cloud cover, ERA5 | `era5_tcc_mean` (total) | % (x 100) | one hue slate, 0–100 |

Grey where a dataset has no value (ERA5-Land over the sea). No colour bars on the small globes; the temperature bar
fades out with the big globe.

The variable count, 294, is computed by `six` from the release files: every column of the latest `series/` file
(static.parquet for ERA5-Static) except `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month` and the `_valid`
share. CPC 4, IMERG 5, CAMS 19, ERA5-Static 16, ERA5-Land 49, ERA5 201; the same as the Climate README's Variables
column. It counts up over 0.6 s and is fully visible for 3.0 s.

Timing for both: flag 7.6 s (fill to 2.56 s, move 2.6–3.4 s, "Major TOM" 3.4–3.8 s, "climate" 4.2–4.6 s, subtitle fully visible 2.5 s), normals 7.5 s; series cut to 4.4 s (drawn in 3.2 s) and
end card to 3.3 s (credits fade in by 0.8 s, fully visible 2.5 s). Render memory raised to 48 GB for the six
arrays per worker.

Live read (`:193`) counts requests and response bytes by wrapping aiohttp, which fsspec's https filesystem uses.

    async def counted_read(self):
        body = await read(self)
        calls["bytes"] += len(body)

## Measurements

- Live read, CPC `by_cell/band=16.parquet`, cell 625U_30L, pandas + fsspec, 2026-11-04: 2.32, 1.69, 1.54 s; median
  1.69 s. 11 requests, 7,082,921 bytes, of a 700,022,318-byte file. 564 rows.
- pyarrow reads the whole row group's column chunks (1,024 cells), so 7.1 MB, not the 0.18 MB the Climate README
  gives for hyparquet with the page index.
- Release sizes by `du -sb`: Climate 5,829.2 GB (ERA5-Monthly 5,172.7 GB with `by_cell/`); Core-Weather 395.4 GB
  (hourly 137.45, daily 257.84, acquisitions 0.08).

## Sample

Loch Lomond cells 621–628U, 26–33L checked in Core-S2L2A metadata (one sample per cell). Candidates with a summer
image of the loch: 625U_30L (2022-08-10) and 623U_30L (2019-09-20). 623U_30L shows Helensburgh and the Clyde, not
the loch. Chose 625U_30L: Loch Lomond and Loch Long, scene cloud cover 0.1 %, ERA5 total cloud cover 2 % at the
image hour.

Stored thumbnail is flat (values 84–118 of 255). `_stretch` (`:183`) maps the 0.5–99.5th percentile of all channels
to 0–255; a per-channel stretch turned clear-fells pink and was dropped.

## v1 ERA5 precipitation bars

Not a data fault. At 624U_29L on 2022-12-11 ERA5 hourly `tp` peaks at 0.017 mm against IMERG 1.55 mm, so bars were
drawn but invisible at that scale. Daily sums agree across ERA5, ERA5-Land, IMERG and CPC on the same cell. Both v2
candidate days have no precipitation in either source; v2 shows cloud cover instead and states "No precipitation in
these 48 hours (ERA5, IMERG)".

## Decisions

- Hand-written orthographic projection; rejected installing cartopy into `miko-torch`.
- Grid fill coloured by column number mod 4; rejected one colour band per quarter (read as stripes again).
- Text at 16 pt or more except the licence line (12 pt): the square shows about 360 px wide on a phone.
- Cloud cover over precipitation for the sample: both summer days are dry.
- Producer names in text, no producer logos (user decision).
- Palette validator not run: `/usr/bin/node` is v10.19, the script needs `??=`. Series colours are the README
  figures' validated slots (`climate_figures.SERIES`, `core_weather_figures.COLOUR`); every line has a direct label.

## Open

- The Climate size (5.83 TB) includes ERA5 `by_cell/`, which is 12 of 20 bands on source.coop (upload job 2426).
  The Climate README still says 3.03 TB for ERA5 (without `by_cell/`).
- Wide layout has empty space below the sources tiles and the access scene.
- No sound.

## Look at first

- screenshots/major-tom-climate-weather-v2-square.mp4
- docs/animations/intro_climate_weather.py:684 (grid scene)
- docs/animations/intro_climate_weather.py:92 (LAYOUTS)
- docs/animations/intro_climate_weather.py:193 (live read)
