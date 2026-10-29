# ==============================================================================
# Reader for Major-TOM/Core-VIIRS-Nighttime-Light (cozip-packaged).
#
# Mirrors the API surface of helpers/ReadAlphaEarth.py: the entry point
# `cell_to_viirs(cell, year, ...)` returns the same shape of result list.
#
# How it works:
#   * INDEX.parquet on the Hub maps each Major TOM tile to (shard, offset, size).
#   * Tiles live as standard GeoTIFFs inside the cozip shards (no decompression
#     needed). We open them in-place via /vsisubfile/<off>_<sz>,/vsicurl/<URL>
#     so GDAL pulls only the bytes for that tile.
# ==============================================================================

# ==============================================================================
# 1. Standard Library Imports
# ==============================================================================
from io import BytesIO

# ==============================================================================
# 2. Third-Party Library Imports
# ==============================================================================
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.cm as cm
from PIL import Image

import rasterio as rio
from huggingface_hub import hf_hub_download

# ==============================================================================
# 3. Dataset configuration & remote index (lazy)
# ==============================================================================
REPO_ID   = "Major-TOM/Core-VIIRS-Nighttime-Light"
REPO_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

# Both the index and the Major TOM grid object are loaded lazily so this
# module's import has zero side effects. The grid is a ~1 GB GeoDataFrame
# instantiated by Grid.__init__; we borrow the instance ReadAlphaEarth
# already created in the Space rather than building a third copy.
_viirs_df = None

def _get_viirs_df():
    global _viirs_df
    if _viirs_df is None:
        path = hf_hub_download(REPO_ID, "INDEX.parquet", repo_type="dataset")
        # pyarrow honors the column projection reliably; pandas via fsspec
        # on a remote URL can hold the whole file in memory.
        table = pq.read_table(path, columns=['id', 'year', 'shard', 'offset', 'size'])
        df = table.to_pandas()
        _viirs_df = df.set_index(['id', 'year']).sort_index()
    return _viirs_df

# ==============================================================================
# 4. Utility Functions
# ==============================================================================

def ntl_to_rgb(arr: np.ndarray, vmin: float = 0.1, vmax: float = 100.0) -> Image.Image:
    """
    Render a single-band VIIRS DNB radiance array as an 8-bit RGB PIL image.

    Uses a fixed absolute log-scale (vmin..vmax in nW/cm²/sr) so dim rural
    tiles render as black and bright urban tiles render as yellow/white —
    consistent across the whole dataset. The 'inferno' colormap matches the
    night-lights aesthetic.
    """
    a = np.asarray(arr, dtype=np.float32)
    a = np.where(np.isfinite(a), a, 0.0)
    a = np.clip(a, vmin, vmax)

    log_min = np.log10(vmin)
    log_max = np.log10(vmax)
    norm = (np.log10(a) - log_min) / (log_max - log_min)
    norm = np.clip(norm, 0.0, 1.0)

    rgba = (cm.inferno(norm) * 255).astype(np.uint8)
    return Image.fromarray(rgba[..., :3])


def cell_to_viirs(cell, year, return_centre=False, return_gridcell=False, return_timestamp=False):
    """
    Render the VIIRS Nighttime Light tile for a Major TOM cell + year.

    Args:
        cell: Major TOM grid id without the level prefix, e.g. "696D_231L".
        year: Annual composite year (int or castable).
        return_centre, return_gridcell, return_timestamp:
            Append the corresponding metadata to the returned list — same
            convention as ReadAlphaEarth.cell_to_aef.

    Returns:
        [PIL.Image, *optional_metadata] on success, or None if the tile
        is not in the index or the read fails.
    """
    row, col = cell.split('_')
    # Reuse the Grid instance ReadAlphaEarth already built — avoids
    # allocating a second multi-GB GeoDataFrame.
    from .ReadAlphaEarth import mt_grid
    lats, lons = mt_grid.rowcol2latlon([row], [col])

    try:
        idx_row = _get_viirs_df().loc[(f'MT10_{cell}', int(year))]
        # If duplicates ever exist, take the first.
        if isinstance(idx_row, pd.DataFrame):
            idx_row = idx_row.iloc[0]

        shard_url = f"{REPO_BASE}/{idx_row['shard']}"
        vsi_path = (
            f"/vsisubfile/{int(idx_row['offset'])}_{int(idx_row['size'])},"
            f"/vsicurl/{shard_url}"
        )

        with rio.Env(
            GDAL_DISABLE_READ_LOCK='YES',
            CPL_VSI_CURL_USE_HEAD='NO',
        ):
            with rio.open(vsi_path) as src:
                arr = src.read(1)

        img = ntl_to_rgb(arr).resize((1068, 1068))

        ret = [img]
        if return_centre:
            ret.append((lats[0], lons[0]))
        if return_gridcell:
            ret.append(cell)
        if return_timestamp:
            ret.append(str(year))
        return ret
    except Exception:
        return None
