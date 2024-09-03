"""
    NOTE: Major TOM standard does not require any specific type of thumbnail to be computed.
    
    Instead these are shared as optional help since this is how the Core dataset thumbnails have been computed.
"""

from rasterio.io import MemoryFile
from PIL import Image
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from matplotlib.colors import LightSource

def get_grayscale(x):
    """
        Normalized grayscale visualisation
    """
    
    # normalize
    x_n = x-x.min()
    x_n = x_n/x_n.max()
    
    return np.uint8(x_n*255)

def get_hillshade(x, azdeg=315, altdeg=45,ve=1):
    """
        Hillshade visualisation for DEM
    """
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    
    return np.uint8(255*ls.hillshade(x, vert_exag=ve))
    
def dem_thumbnail(dem, dem_NODATA = -32768.0, hillshade=True):
    """
        Takes vv and vh numpy arrays along with the corresponding NODATA values (default is -32768.0)
        
        Returns a numpy array with the thumbnail
    """
    if hillshade:
        return get_hillshade(dem)
    else:
        return get_grayscale(dem)
    

def dem_thumbnail_from_datarow(datarow):
    """
        Takes a datarow directly from one of the data parquet files
        
        Returns a PIL Image
    """

    with MemoryFile(datarow['DEM'][0].as_py()) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            dem=f.read().squeeze()
            dem_NODATA = f.nodata

    img = dem_thumbnail(dem, dem_NODATA)

    return Image.fromarray(img,'L')

if __name__ == '__main__':    
    from fsspec.parquet import open_parquet_file
    import pyarrow.parquet as pq

    print('[example run] reading file from HuggingFace...')
    url = "https://huggingface.co/datasets/Major-TOM/Core-DEM/resolve/main/images/part_01001.parquet"
    with open_parquet_file(url) as f:
        with pq.ParquetFile(f) as pf:
            first_row_group = pf.read_row_group(1)
    
    print('[example run] computing the thumbnail...')    
    thumbnail = dem_thumbnail_from_datarow(first_row_group)

    thumbnail_fname = 'example_thumbnail.png'
    thumbnail.save(thumbnail_fname, format = 'PNG')
    print('[example run] saved as "{}"'.format(thumbnail_fname))