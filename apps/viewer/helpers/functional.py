from fsspec.parquet import open_parquet_file 
import fsspec
import pyarrow.parquet as pq
from .grid import *
from .ReadAlphaEarth import *
from .ReadVIIRSNTL import *
import pandas as pd
from io import BytesIO
import os
from PIL import Image
import datetime

# GLOBAL VARIABLES
if os.path.isfile('helpers/s2l2a_metadata.parquet'):
    l2a_meta_path = 'helpers/s2l2a_metadata.parquet'
else:
    DATASET_NAME = 'Major-TOM/Core-S2L2A'
    l2a_meta_path = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet'.format(DATASET_NAME)

if os.path.isfile('helpers/s2l1c_metadata.parquet'):
    l1c_meta_path = 'helpers/s2l1c_metadata.parquet'
else:
    DATASET_NAME = 'Major-TOM/Core-S2L1C'
    l1c_meta_path = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet'.format(DATASET_NAME)

if os.path.isfile('helpers/s1rtc_metadata.parquet'):
    rtc_meta_path = 'helpers/s1rtc_metadata.parquet'
else:
    DATASET_NAME = 'Major-TOM/Core-S1RTC'
    rtc_meta_path = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet'.format(DATASET_NAME)

if os.path.isfile('helpers/dem_metadata.parquet'):
    dem_meta_path = 'helpers/dem_metadata.parquet'
else:
    DATASET_NAME = 'Major-TOM/Core-DEM'
    dem_meta_path = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet'.format(DATASET_NAME)

grid = Grid(10, latitude_range=(-90,90), longitude_range=(-180,180))
l2a_df = pd.read_parquet(l2a_meta_path)
l1c_df = pd.read_parquet(l1c_meta_path)
rtc_df = pd.read_parquet(rtc_meta_path)
dem_df = pd.read_parquet(dem_meta_path)

df_dict = {
    'Sentinel-2 L2A' : l2a_df,
    'Sentinel-2 L1C' : l1c_df,
    'Sentinel-1 RTC' : rtc_df,
    'COP-DEM' : dem_df
}

'''
    --- SigLIP Search ---
'''
from .TextSearch import *

BASE_DIR='./'
INDEX_OUTPUT = f'{BASE_DIR}/siglip_ivfpq.index'
METADATA_OUTPUT = f'{BASE_DIR}/siglip_ivfpq_metadata.parquet'

search = SearchSigLIP(index_path=INDEX_OUTPUT,
                      metadata_path=METADATA_OUTPUT)

'''
    --- Helper Functions ---
'''

def pretty_date(input):
    template = '%Y%m%dT%H%M%S' if 'T' in input else '%Y%m%d%H%M%S'
    return datetime.datetime.strptime(input, template).strftime('%H:%M:%S - %d %b %Y')

# HELPER FUNCTIONS
def gridcell2ints(grid_string):
    up = int(grid_string.split('_')[0][:-1]) * (2*int(grid_string.split('_')[0][-1]=='U') - 1) # +ve if up
    right = int(grid_string.split('_')[1][:-1]) * (2*int(grid_string.split('_')[1][-1]=='R') - 1) # +ve if R

    return up, right

def row2image(parquet_url, parquet_row, fullrow_read=True):

    if fullrow_read:
        # option 1
        f=fsspec.open(parquet_url)
        temp_path = f.open()
    else:
        # option 2
        temp_path = open_parquet_file(parquet_url,columns = ["thumbnail"])
    
    with pq.ParquetFile(temp_path) as pf:
        first_row_group = pf.read_row_group(parquet_row, columns=['thumbnail'])

    stream = BytesIO(first_row_group['thumbnail'][0].as_py())
    return Image.open(stream)

def row2s2(parquet_url, parquet_row, s2_bands = ["B04", "B03", "B02"]):
    with open_parquet_file(parquet_url,columns = s2_bands) as f:
        with pq.ParquetFile(f) as pf:
            first_row_group = pf.read_row_group(parquet_row, columns=s2_bands)

    return first_row_group

def cell2row(grid_string, meta_df, return_row = False):
    row_U, col_R = gridcell2ints(grid_string)
    R = meta_df.query('grid_row_u == {} & grid_col_r == {}'.format(row_U, col_R))

    if not R.empty:
        if return_row:
            return R.parquet_url.item(), R.parquet_row.item(), R
        else:
            return R.parquet_url.item(), R.parquet_row.item()
    else:
        return None
        
def map_to_image(map, return_centre=False, return_gridcell=False, return_timestamp=False, source='Sentinel-2 L2A'):

    try:
        # 1. get bounds
        bbox = map.get_bbox()
        center = [(bbox[3]+bbox[1])/2, (bbox[2]+bbox[0])/2]
    except:
        return None

    # 2. translate coordinate to major-tom tile
    rows, cols = grid.latlon2rowcol([center[0]], [center[1]])

    return cell_to_image(cell="{}_{}".format(rows[0],cols[0]),
                         return_centre=return_centre,
                         return_gridcell=return_gridcell,
                         return_timestamp=return_timestamp,
                         source=source)

def text_to_image(text, return_centre=False, return_gridcell=False, return_timestamp=False, source='Sentinel-2 L2A'):

    cell = search.faiss(text)[0]["grid_cell"]

    return cell_to_image(cell=cell,
                         return_centre=return_centre,
                         return_gridcell=return_gridcell,
                         return_timestamp=return_timestamp,
                         source=source)
    

def cell_to_image(cell, return_centre=False, return_gridcell=False, return_timestamp=False, source='Sentinel-2 L2A'):

    try:
        df = df_dict[source]
    except:
        # fallback to default (e.g. for metadata for AEF)
        df = df_dict['Sentinel-2 L2A']
        
    row = cell2row(cell, df, return_row = True)
    
    if row is not None:
        parquet_url, parquet_row, meta_row = row

        # AEF is special - it's read live
        if source == 'AlphaEarth Embeddings':
            year = int(meta_row.iloc[0].timestamp[:4])
            return cell_to_aef(cell, year, return_centre=return_centre, return_gridcell=return_gridcell, return_timestamp=return_timestamp)
        # VIIRS NTL is also read live from cozip
        if source == 'VIIRS Nighttime Light':
            year = int(meta_row.iloc[0].timestamp[:4])
            return cell_to_viirs(cell, year, return_centre=return_centre, return_gridcell=return_gridcell, return_timestamp=return_timestamp)

        try:
            img = row2image(parquet_url, parquet_row)
        except:
            return None
            
        # 4. acquire image # X. update map
        lat, lon = meta_row.centre_lat.item(), meta_row.centre_lon.item()

        ret = [img.resize((1068,1068))]
        if return_centre:
            ret.append((lat,lon))
        if return_gridcell:
            ret.append(meta_row.grid_cell.item())
        if return_timestamp:
            if 'timestamp' in meta_row.columns:
                ret.append(pretty_date(meta_row.timestamp.item()))
            else:
                ret.append('Not Available')
            
        return ret
    else:
        return None