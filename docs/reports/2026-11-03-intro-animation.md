> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-11-03 — Intro animation, Major TOM Climate and Core-Weather

One MP4 for LinkedIn: 1080 × 1080, 30 fps, 45.0 s, H.264 yuv420p, silent, 3.8 MB (3,755,281 bytes, 1,350 frames).
Output in `/data/Major-TOM/animations/intro/major-tom-climate-weather-1080.mp4`, not in git. Key frames in
`screenshots/intro-*.png` (git-ignored). Generator `docs/animations/intro_climate_weather.py`, uncommitted.

## Scenes

| Time (s) | Shows | Data |
|---|---|---|
| 0–5.3 | Major TOM Climate logo: 16 tiles appear column by column, wordmark wiped in, one line of text | `docs/brand/mt-climate.svg` |
| 5.3–15.3 | ERA5 2 m temperature normals, map fills in as patches, then January to December | `ERA5-Monthly/climatology_overviews/MM.parquet` (4 × 4-cell blocks), local |
| 15.3–27.3 | Monthly precipitation 2014–2023 at Loch Lomond, Corvara, Manaus; CPC, IMERG, ERA5 drawn month by month | `by_cell/band=NN.parquet` of the three datasets, local |
| 27.3–33.3 | Loch Lomond Sentinel-2 L2A sample (2022-12-11 11:45 UTC) and its hourly temperature and precipitation, ±24 h | Core-S2L2A metadata and thumbnail (HF, one row group); Core-Weather `merged/hourly` |
| 33.3–39.3 | ERA5 2 m temperature at acquisition time, filled in by acquisition date, 2015-12 to 2024-02 | `merged/hourly`, step 0: 3,704,013 acquisitions, 2,238,689 cells, 2,976 day files |
| 39.3–45 | Logo, dataset names and source.coop addresses, Earth Compress and Asterisk Labs logos, credit, Copernicus attribution | `docs/brand/earth-compress.svg`, `apps/climate-globe/asterisk-mark.svg` |

Each scene fades in and out through the page colour over 0.3 s. The last frame holds.

## docs/animations/intro_climate_weather.py

Three steps, two environments: `logos` (python-last, Chromium), `prep` and `render` (miko-torch, Basemap).
All three ran through `srun`; prep took one 16-CPU job, render 40 s on 16 CPUs.

The SVGs use path data, which matplotlib cannot read. Chromium rasterises them once; the tiles stay vector,
parsed from the SVG's `<polygon>` elements, so they can be animated one by one (`:64`).

    svg = LOGO_SVG.read_text()
    wordmark = re.sub(r"<polygon[^>]*/>", "", svg)                      # the tiles are drawn by the animation

Temperature at acquisition is accumulated in time on 12 × 12-cell blocks, 150 checkpoints between the first and
last acquisition (`:163`). The last checkpoint is the README figure `temperature-at-acquisition.png`.

    checkpoints = np.linspace(times[0].astype("int64"), times[-1].astype("int64"), 150).astype("datetime64[ns]")
    ends = np.searchsorted(times, checkpoints, side="right")
    ...
        sums += np.bincount(block[start:end], values[start:end], minlength=sums.size)
        counts += np.bincount(block[start:end], minlength=counts.size)

The normals map fills in along a smoothed random field, ranked so the filled share grows linearly (`:236`).
Uniform noise looked like static.

    field = gaussian_filter(rng.random(self.share4.shape), 24, mode="wrap")
    ...
    self.reveal4 = np.argsort(np.argsort(field, axis=None)).reshape(field.shape) / field.size

Map pictures are drawn at block resolution with the README colour scale (`COLD_HOT`, −40 to +40 °C, grey at 0) and
Basemap coastline segments saved once in `coast.npz`. Basemap is not called per frame.

Unit conversions as in `docs/evidence/climate_figures.py`: CPC monthly total as stored; IMERG mm/h × 24 × days in the
month; ERA5 `era5_tp` m/day × 1000 × days (`:97`).

## Reused

- `climate_figures.py`: `use_major_tom_font` (IBM Plex Mono), `PLACES`, `inside_earth`, `colour_bar`, `COLD_HOT`, `PAGE`, `INK`.
- `core_weather_figures.py`: `thumbnail`, `weather_rows`, `_style`, `COLOUR`.
- `style.py`: `ease`, `phase`, `mix`; League Spartan from `docs/animations/fonts` for the "asterisk labs" wordmark.

## Checks

- Series colours `COLOUR` (ERA5 #2a78d6, IMERG #eb6834, CPC #1baf7a): dataviz validator in Chromium (Node 10 on the
  server cannot run it), all pairs, light surface: PASS on lightness, chroma, CVD (worst ΔE 9.2) and normal vision
  (24.0). WARN: CPC green 2.74:1 against the page. Relief: legend row plus direct labels at the line ends.
- Colour follows the source in every scene: ERA5 blue, IMERG orange, CPC green, ERA5-Land grey.
- Frames checked by eye at 1.2, 2.6, 4.5, 6.4, 12, 18.5, 20, 26, 29.5, 32, 35, 38.5 and 44.5 s.
- ffprobe: h264, 1080 × 1080, yuv420p, 30/1, 1,350 frames, 45.000 s.

## Decisions

- IBM Plex Mono for all data text, as the README figures; League Spartan only in the Asterisk wordmark.
- Square 1080 × 1080 only. Rejected a 1920 × 1080 copy: every layout is placed in square pixels, a wide version
  needs its own layouts.
- Temperature for the global maps, not precipitation: one diverging scale for both map scenes.
- Fill of the acquisition map by date, not at random: the dates are real data.
- Loch Lomond for the sample: 0 % cloud in the Core-S2L2A row. The winter thumbnail is dark (low sun, December).
- No GIF preview: the MP4 is 3.8 MB.

## Open

- Major TOM Climate on source.coop is Unlisted and ERA5 `by_cell/` is still uploading; the end card names its address.
- The Copernicus line says "information 2026" (ERA5-Monthly README) and "Sentinel data 2022" (the thumbnail year).
- No sound; LinkedIn plays muted by default.

## Look at first

- screenshots/intro-sample.png, screenshots/intro-end.png
- docs/animations/intro_climate_weather.py:391 (scene_series)
- /data/Major-TOM/animations/intro/major-tom-climate-weather-1080.mp4
