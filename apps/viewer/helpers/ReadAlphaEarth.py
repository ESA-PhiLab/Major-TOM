# ==============================================================================
# 1. Standard Library Imports
# ==============================================================================
import json
import os
from io import BytesIO

# ==============================================================================
# 2. Third-Party Library Imports
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from affine import Affine # Recommended for working with rasterio transforms
from shapely.geometry import Polygon, Point
import pyproj
from sklearn.decomposition import PCA
from PIL import Image

# Geospatial/Cloud Tools
import rasterio as rio
from rasterio.windows import from_bounds
import s3fs
from fsspec.parquet import open_parquet_file
import pyarrow.parquet as pq

from .grid import *

fs = s3fs.S3FileSystem(anon=True)
print('[DONE]\nReading AEF index from remote location...')
aef_df = gpd.read_parquet('s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet')
mt_grid = Grid(10, latitude_range=(-90,90), longitude_range=(-180,180))

# ==============================================================================
# 3. Utility Functions
# ==============================================================================

def row2thumbnail(row):
    '''
        Read the thumbnail column from a Major TOM metadata parquet row
    '''

    with open_parquet_file(row.parquet_url,columns = ["thumbnail"]) as f:
        with pq.ParquetFile(f) as pf:
            first_row_group = pf.read_row_group(row.parquet_row, columns=['thumbnail'])

    stream = BytesIO(first_row_group['thumbnail'][0].as_py())
    return Image.open(stream)

def rgb_pca(arr: np.ndarray) -> Image.Image:
    """
    Applies PCA to a 3D array to reduce its dimensions to 3,
    then converts it into a PIL Image.

    Args:
        arr (np.ndarray): Input array of shape (H, W, C).

    Returns:
        Image.Image: A PIL Image object ready to be saved.
    """
    h,w,c=arr.shape

    # Ensure the input array has the correct shape
    if len(arr.shape) != 3:
        raise ValueError("Input array must be of shape (Height, Width, Channels)")

    # Reshape the array for PCA
    reshaped_arr = arr.reshape(-1, c)

    # Initialize and fit PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    pca.fit(reshaped_arr)

    # Transform the data using the fitted PCA
    transformed_arr = pca.transform(reshaped_arr)

    # Normalize the data to the 0-255 range for image conversion
    # Min-Max scaling for each principal component
    normalized_arr = np.zeros_like(transformed_arr)
    for i in range(transformed_arr.shape[1]):
        min_val = transformed_arr[:, i].min()
        max_val = transformed_arr[:, i].max()
        normalized_arr[:, i] = 255 * (transformed_arr[:, i] - min_val) / (max_val - min_val)

    # Reshape the array back to its original dimensions (h, w, 3)
    # and convert to an 8-bit unsigned integer type
    rgb_image_data = (normalized_arr).reshape(h, w, 3).astype(np.uint8)

    # Create a PIL Image from the processed NumPy array
    #return arr
    pil_image = Image.fromarray(rgb_image_data*(arr.sum(-1)!=c*65535)[:,:,None])

    return pil_image

def get_product_window(lat, lon, utm_zone=4326, mt_grid_dist = 10, box_size = 10680):
    """
        Takes a reference coordinate for top-left corner (lat, lon) of a Major TOM cell
        and returns a product footprint for a product in the specified utm_zone (needs to be extracted from a given product)


        mt_grid_dist (km) : distance of a given Major TOM grid (10 km is the default)
        box_size (m) : length
    """
    # offset distributed evenly on both sides
    box_offset = (box_size-mt_grid_dist*1000)/2 # metres

    if isinstance(utm_zone, int):
        utm_crs = f'EPSG:{utm_zone}'
    else:
        utm_crs = utm_zone

    # Define transform
    transformer = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)

    # Get corners in UTM coordinates
    left,bottom = transformer.transform(lon, lat)
    left,bottom = left-box_offset, bottom-box_offset
    right,top = left+box_size,bottom+box_size

    utm_footprint = Polygon([
        (left,bottom),
        (right,bottom),
        (right,top),
        (left,top)
    ])

    return utm_footprint, utm_crs

def cell_to_aef(cell, year, return_centre=False, return_gridcell=False, return_timestamp=False):

    row, col = cell.split('_')
    
    # 1. define major tom grid cell
    lats, lons = mt_grid.rowcol2latlon([row],[col])

    try:
        # 2. find the right COG
        aef_row = aef_df[aef_df.contains(Point(lons,lats)) & (aef_df.year == year)].iloc[0]
        
        # 3. map lat-lon to utm footprint
        window, proj = get_product_window(lats[0], lons[0], utm_zone=aef_row.crs)
    
        # Set the environment variable for GDAL to enable S3 access over HTTPS
        # This is crucial for reading COGs directly from S3.
        with rio.Env(
            aws_unsigned=True,
            aws_no_sign_request=True,
            GDAL_DISABLE_READ_LOCK='YES',
            CPL_VSI_CURL_USE_HEAD='NO'
        ):
            with rio.open(aef_row.path) as src:
                minx, miny, maxx, maxy = window.bounds
                src_window = from_bounds(minx, maxy, maxx, miny, src.transform) # CAREFUL - COGs here are non-standard (positive y-dir, so require reordering)
                arr = src.read(window=src_window).transpose(1,2,0)[::-1,...]
    
        # PCA
        pca_img = rgb_pca(arr)
    
        ret = [pca_img.resize((1068,1068))]
        if return_centre:
            ret.append((lats[0],lons[0]))
        if return_gridcell:
            ret.append(cell)
        if return_timestamp:
            ret.append(str(year))
    
        return ret
    except:
        return None