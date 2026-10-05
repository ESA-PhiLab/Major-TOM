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
    """
    Crops the given footprint to the specified bounding box.

    Args:
        footprint (shapely.geometry.Polygon): The original footprint of the image or area.
        height (int): Height of the image (in pixels).
        width (int): Width of the image (in pixels).
        crop_bbox (list): The bounding box to crop the footprint. The format is 
                          [col_start, row_start, col_end, row_end], where:
                          - col_start, row_start: top-left corner
                          - col_end, row_end: bottom-right corner

    Returns:
        shapely.geometry.Polygon: The cropped bounding box in the same coordinate reference system (CRS) as the original footprint.
    """
    
    transform = from_bounds(*footprint.bounds, width, height)

    # Convert pixel coordinates (col, row) to spatial coordinates (e.g., UTM)
    # Using the raster's affine transform
    min_x, min_y = transform * (crop_bbox[0], crop_bbox[1])  # (col_start, row_start)
    max_x, max_y = transform * (crop_bbox[2], crop_bbox[3])  # (col_end, row_end)

    # Create a Shapely polygon for the crop's bounding box in UTM
    return box(min_x, min_y, max_x, max_y)

def fragment_unfold(image,fragment_size,overlap):
    """
    Unfold operation for a fragment with overlap. This function extracts image patches (fragments) with a specified 
    size and overlap between them.

    Args:
        image (torch.Tensor or np.ndarray): The input image to be fragmented (height, width, channels).
        fragment_size (int or list): The size of each fragment. Can be a single integer for square fragments or 
                                     a list of two integers for non-square fragments.
        overlap (int or list): The overlap between adjacent fragments. Can be a single integer or a list of two integers.

    Returns:
        torch.Tensor: The unfolded fragments of the image, each with the specified size and overlap.
    """
    
    # Convert image to a tensor and reorder dimensions if necessary
    if not torch.is_tensor(image):
        image = torch.from_numpy(image).permute(2, 0, 1)  # Rearrange to (channels, height, width)
    if len(image.shape) < 4:
        image = image.unsqueeze(0)  # Add batch dimension

    b, c, h, w = image.shape

    # Ensure fragment size is a list
    if isinstance(fragment_size, int):
        fragment_size = [fragment_size, fragment_size]
    if isinstance(overlap, int):
        overlap = [overlap, overlap]

    # Calculate stride based on fragment size and overlap
    stride = [f - o for f, o in zip(fragment_size, overlap)]

    # Perform the unfolding operation
    uf = torch.nn.functional.unfold(image, fragment_size, dilation=1, padding=0, stride=stride)

    # Reshape and permute to return the unfolded image fragments
    return uf.view(b, c, *fragment_size, -1).permute(0, 4, 1, 2, 3)[0]

def fragment_fn(img,
                fragment_size,
                target_overlap,
                border_shift=True, # determines whether the outer border is shifted to ensure full coverage
                return_indices=False,
                verbose=False
               ):
    """
    Fragment an image into smaller patches with a specified fragment size and overlap.

    This function handles different scenarios based on image size, fragment size, and overlap, 
    and creates fragments from the input image accordingly. It also supports shifting the outer 
    border of fragments to ensure full coverage of the image.

    Args:
        img (np.ndarray or torch.Tensor): The input image to be fragmented (height, width, channels).
        fragment_size (int or list): The size of the fragments. Can be a single integer (square) or a list of two integers (non-square).
        target_overlap (float): The target overlap between adjacent fragments, in pixels.
        border_shift (bool): Whether to shift the border of fragments to ensure full coverage of the image. Default is True.
        return_indices (bool): If True, the function will also return the indices (offsets) for each fragment. Default is False.
        verbose (bool): If True, the function will print additional details about the overlap. Default is False.

    Returns:
        torch.Tensor or tuple: 
            - If `return_indices` is False, a tensor containing the image fragments.
            - If `return_indices` is True, a tuple of the image fragments and their offsets.
    """

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