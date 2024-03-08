from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot(sample, bands = ['B04', 'B03', 'B02'], scaling=2e3):
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
