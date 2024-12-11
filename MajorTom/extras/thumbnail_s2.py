"""
    NOTE: Major TOM standard does not require any specific type of thumbnail to be computed.
    
    Instead these are shared as optional help since this is how the Core dataset thumbnails have been computed.
"""

from rasterio.io import MemoryFile
from PIL import Image
import numpy as np

def s2l2a_thumbnail(B04, B03, B02, gain=1.3, gamma=0.6):
    """
        Takes B04, B03, B02 numpy arrays along with the corresponding NODATA values (default is -32768.0)
        
        Returns a numpy array with the thumbnail
    """
    
    # concatenate
    thumb = np.stack([B04, B03, B02], -1)

    # apply gain & gamma
    thumb = gain*((thumb/10_000)**gamma)
    
    return (thumb.clip(0,1)*255).astype(np.uint8)

def s2l2a_thumbnail_from_datarow(datarow):
    """
        Takes a datarow directly from one of the data parquet files
        
        Returns a PIL Image
    """

    # red
    with MemoryFile(datarow['B04'][0].as_py()) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            B04=f.read().squeeze()
            B04_NODATA = f.nodata

    # green
    with MemoryFile(datarow['B03'][0].as_py()) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            B03=f.read().squeeze()
            B03_NODATA = f.nodata

    # blue
    with MemoryFile(datarow['B02'][0].as_py()) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            B02=f.read().squeeze()
            B02_NODATA = f.nodata

    img = s2l2a_thumbnail(B04,B03,B02)

    return Image.fromarray(img)

if __name__ == '__main__':  
    from fsspec.parquet import open_parquet_file
    import pyarrow.parquet as pq

    print('[example run] reading file from HuggingFace...')
    url = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/images/part_01000.parquet"
    with open_parquet_file(url, columns = ["B04", "B03", "B02"]) as f:
        with pq.ParquetFile(f) as pf:
            first_row_group = pf.read_row_group(1, columns = ["B04", "B03", "B02"])
    
    print('[example run] computing the thumbnail...')    
    thumbnail = s2l2a_thumbnail_from_datarow(first_row_group)
    
    thumbnail.save('example_thumbnail.png', format = 'PNG')