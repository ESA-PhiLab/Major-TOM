"""
    NOTE: Major TOM standard does not require any specific type of thumbnail to be computed.
    
    Instead these are shared as optional help since this is how the Core dataset thumbnails have been computed.
"""

from rasterio.io import MemoryFile
from PIL import Image
import numpy as np

def s1rtc_thumbnail(vv, vh, vv_NODATA = -32768.0, vh_NODATA = -32768.0):
    """
        Takes vv and vh numpy arrays along with the corresponding NODATA values (default is -32768.0)
        
        Returns a numpy array with the thumbnail
    """
    
    # valid data masks
    vv_mask = vv != vv_NODATA
    vh_mask = vh != vh_NODATA

    # remove invalid values before log op
    vv[vv<0] = vv[vv>=0].min()
    vh[vh<0] = vh[vh>=0].min()

    # apply log op
    vv_dB = 10*np.log10(vv)
    vh_dB = 10*np.log10(vh)

    # scale to 0-255
    vv_dB = (vv_dB - vv_dB[vv_mask].min()) / (vv_dB[vv_mask].max() - vv_dB[vv_mask].min()) * 255
    vh_dB = (vh_dB - vh_dB[vh_mask].min()) / (vh_dB[vh_mask].max() - vh_dB[vh_mask].min()) * 255

    # represent nodata as 0
    vv_dB[vv_mask==0] = 0
    vh_dB[vh_mask==0] = 0

    # false colour composite
    return np.stack([vv_dB,
                    255*(vv_dB+vh_dB)/np.max(vv_dB+vh_dB),
                    vh_dB
                   ],-1).astype(np.uint8)

def s1rtc_thumbnail_from_datarow(datarow):
    """
        Takes a datarow directly from one of the data parquet files
        
        Returns a PIL Image
    """

    with MemoryFile(datarow['vv'][0].as_py()) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            vv=f.read().squeeze()
            vv_NODATA = f.nodata
    
    with MemoryFile(datarow['vh'][0].as_py()) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            vh=f.read().squeeze()
            vh_NODATA = f.nodata

    img = s1rtc_thumbnail(vv, vh, vv_NODATA=vv_NODATA, vh_NODATA=vh_NODATA)

    return Image.fromarray(img)

if __name__ == '__main__':    
    from fsspec.parquet import open_parquet_file
    import pyarrow.parquet as pq

    print('[example run] reading file from HuggingFace...')
    url = "https://huggingface.co/datasets/Major-TOM/Core-S1RTC/resolve/main/images/part_00001.parquet"
    with open_parquet_file(url) as f:
        with pq.ParquetFile(f) as pf:
            first_row_group = pf.read_row_group(1)
    
    print('[example run] computing the thumbnail...')    
    thumbnail = s1rtc_thumbnail_from_datarow(first_row_group)

    thumbnail_fname = 'example_thumbnail.png'
    thumbnail.save(thumbnail_fname, format = 'PNG')
    print('[example run] saved as "{}"'.format(thumbnail_fname))