from rasterio.io import MemoryFile
import numpy as np
from PIL import Image
from io import BytesIO

def plot(sample, bands = ['B04', 'B03', 'B02'], scaling=2e3):
    import matplotlib.pyplot as plt  # [viz] extra; only plotting needs it
    img = []
    for b in bands:
        img.append(read_tif_bytes(sample[b]))
    plt.imshow(np.stack(img, -1)/2e3)

def read_tif_bytes(tif_bytes):
    with MemoryFile(tif_bytes) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            return f.read().squeeze()

def read_png_bytes(png_bytes):
    stream = BytesIO(png_bytes)
    return Image.open(stream)


def read_tif(tif_bytes, dataset: str | None = None, raw: bool = False):
    """(array, transform, crs) of a sample's GeoTIFF bytes.

    For Core samples (`dataset`, e.g. "Core-S2L2A"), the stored transform is corrected to the grid the pixels
    were read on (majortom.spec.correct_core_transform; docs/SPEC.md 7.2) unless `raw` is True.
    """
    from .spec import correct_core_transform

    with MemoryFile(tif_bytes) as mem_f:
        with mem_f.open(driver="GTiff") as f:
            data, transform, crs = f.read().squeeze(), f.transform, f.crs
    if dataset is not None and not raw:
        transform = correct_core_transform(transform, dataset, crs)
    return data, transform, crs
