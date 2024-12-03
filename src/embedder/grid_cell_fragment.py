import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.ops import transform
from pyproj import CRS, Transformer
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
from rasterio.transform import from_bounds, xy
#from rasterio.windows import Window, from_bounds
import rasterio as rio

def crop_footprint(footprint, height, width, crop_bbox):
    # Define as: crop_bbox = [col_start, row_start, col_end, row_end]  # [min_col, min_row, max_col, max_row]
    
    transform = from_bounds(*footprint.bounds, width, height)

    # Convert pixel coordinates (col, row) to spatial coordinates (e.g., UTM)
    # Using the raster's affine transform
    min_x, min_y = transform * (crop_bbox[0], crop_bbox[1])  # (col_start, row_start)
    max_x, max_y = transform * (crop_bbox[2], crop_bbox[3])  # (col_end, row_end)

    # Create a Shapely polygon for the crop's bounding box in UTM
    return box(min_x, min_y, max_x, max_y)

def fragment_unfold(image,fragment_size,overlap):
    '''
        Unfold operation for a fragment with overlap
    '''
    if not torch.is_tensor(image):
        image = torch.from_numpy(image).permute(2,0,1)
    if len(image.shape) < 4:
        image = image.unsqueeze(0)

    b,c,h,w=image.shape

    if isinstance(fragment_size,int):
        fragment_size=[fragment_size,fragment_size]
    if isinstance(overlap,int):
        overlap=[overlap,overlap]

    stride = [f-o for f,o in zip(fragment_size,overlap)]
    #print('STRIDE {}'.format(stride))
    uf = torch.nn.functional.unfold(image, fragment_size, dilation=1, padding=0, stride=stride)
    return uf.view(b,c,*fragment_size,-1).permute(0,4,1,2,3)[0]

def fragment_fn(img,
                fragment_size,
                target_overlap,
                border_shift=True, # determines whether the outer border is shifted to ensure full coverage
                return_indices=False,
                verbose=False
               ):
    '''
        End-to-end: fragmenting function - ATTENTION: SQUARE IMAGES AND FRAGMENTS ONLY

        It adapts to various img sizes, fragment_sizes and desired overlap (in pixels)

        Scenario 1:
            fragment_size < img_size : (not supported)
        Scenario 2:
            fragment_size == img_size : return image
        Scenario 3:
            fragment_size*2 - overlap < img_size : return only 2 fragments aligned to borders
        Scenario 4: (most common):
            fragment_size*2 - overlap > img_size : assign border fragments and fill the middle with other fragments if there is enough space (distributed evenly)
    '''

    h,w,c=img.shape

    assert h==w # SQUARE IMAGES SUPPORT ONLY
    
    hf, wf = fragment_size, fragment_size
    ho, wo = target_overlap*hf, target_overlap*wf

    assert h >= hf and w >= wf # reject Scenario 1

    # Scenario 2
    if h == hf or w == wf:
        if not torch.is_tensor(img):
            img=torch.from_numpy(img).permute(2,0,1)
        return img.view(1,1,c,h,w)

    # Scenario 3 & 4
    
    # determine number of segments between the centers of outermost fragments
    h_n = max(1, int(np.round((h-hf)/(hf-ho))))
    w_n = max(1, int(np.round((w-wf)/(wf-wo))))
    
    # adjust practical overlap (divide the distance between the centers of outermost fragments by the true number of segments)
    aho = int(np.ceil(hf-(h-hf)/(h_n)))
    awo = int(np.ceil(wf-(w-wf)/(w_n)))
    
    # compute fragments (might not exactly fill the outermost border)
    topleft = fragment_unfold(img.permute(2,0,1),fragment_size=(hf,wf), overlap=(aho,awo)).view(1+h_n, 1+w_n, c, hf, wf)

    full = topleft

    if border_shift:
    
        if  h > hf+h_n*(hf-aho) or w > wf+w_n*(wf-awo):
            #print('Outers...')
            bottomleft = fragment_unfold(img[-hf:,:,:],fragment_size=(hf,wf), overlap=(aho,awo)).view(1,1+w_n,c,hf,wf)
            topright = fragment_unfold(img[:,-wf:,:],fragment_size=(hf,wf), overlap=(aho,awo)).view(1+h_n,1,c,hf,wf)

            # Shift last row and col to the border of the original
            full[:,-1,None] = topright
            full[-1] = bottomleft

    if verbose:
        print('Target Overlap: {} pixels. Feasible Overlap: {} pixels.'.format(ho,aho))

    if not return_indices:
        return full
    else:
        offset=-1*torch.ones(*full.shape[:2],2)
        for ridx in range(full.shape[0]):
            for cidx in range(full.shape[1]):
                offset[ridx,cidx,1] = cidx * (hf-aho)
                offset[ridx,cidx,0] = ridx * (wf-awo)

                if border_shift:
                    offset[ridx,-1,1] = h-hf
                    offset[-1,cidx,0] = w-wf

        return full,offset
                

# def get_product_window(lat, lon, utm_zone=4326, mt_grid_dist = 10, box_size = 10680):
#     """
#         Takes a reference coordinate for top-left corner (lat, lon) of a Major TOM cell
#         and returns a product footprint for a product in the specified utm_zone (needs to be extracted from a given product)


#         mt_grid_dist (km) : distance of a given Major TOM grid (10 km is the default)
#         box_size (m) : length 
#     """
#     # offset distributed evenly on both sides
#     box_offset = (box_size-mt_grid_dist*1000)/2 # metres

#     if isinstance(utm_zone, int):
#         utm_crs = f'EPSG:{utm_zone}'
#     else:
#         utm_crs = utm_zone
    
#     # Define transform
#     transformer = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)

#     # Get corners in UTM coordinates
#     left,bottom = transformer.transform(lon, lat)
#     left,bottom = left-box_offset, bottom-box_offset
#     right,top = left+box_size,bottom+box_size

#     utm_footprint = Polygon([
#         (left,bottom),
#         (right,bottom),
#         (right,top),
#         (left,top)
#     ])

#     return utm_footprint, utm_crs