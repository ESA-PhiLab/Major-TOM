import numpy as np
import geopandas as gpd
import hashlib
from rasterio.io import MemoryFile

from grid_cell_fragment import *
from models import *
import cv2

class MajorTOM_Embedder(torch.nn.Module):

    def __init__(self, embedder, target_overlap=0.1, border_shift=True):
        super().__init__()

        # Model
        self.embedder = embedder

        # Fragmentation Settings
        self.frag_params = params = {
            'fragment_size' : self.embedder.size[0],
            'target_overlap' : target_overlap,
            'border_shift' : border_shift
        }

        # Data types for the output dataframe (commented columns need no conversion)
        self.column_types = {
            #'unique_id' :,
            #'embedding' : ,
            #'timestamp' : ,
            #'product_id' : ,
            #'grid_cell' : ,
            'grid_row_u' : 'int16',
            'grid_col_r' : 'int16',
            'centre_lat' : 'float32',
            'centre_lon' : 'float32',
            #'utm_footprint' : ,
            #'utm_crs' : ,
            #'pixel_bbox' : ,            
        }

    def bands(self):
        '''
            Return set of input bands (in correct order)
        '''
        return self.embedder.bands

    def size(self):
        '''
            Return input image size
        '''
        return self.embedder.size

    def calculate_checksum(self,geometry, timestamp, product_id, embedding):
        combined = f"{geometry}_{timestamp}_{product_id}_{embedding}"
        checksum = hashlib.sha256(combined.encode()).hexdigest()
        return checksum

    def _read_image(self, row):

        # Read the file
        img = []
        for band in self.embedder.bands:
            with MemoryFile(row[band][0].as_py()) as mem_f:
                with mem_f.open(driver='GTiff') as f:
                    crs = f.crs
                    footprint = box(*f.bounds)
                    img.append(f.read()[0])

        # optional upsampling
        shapes = [layer.shape for layer in img]
        if any([el!=shapes[0] for el in shapes]): # if any resolution mismatch
            h, w = max([el[0] for el in shapes]), max([el[1] for el in shapes]) # maximum size
            for layer_idx, layer in enumerate(img):
                if layer.shape != (h,w):
                    img[layer_idx] = cv2.resize(layer, (h,w), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(np.stack(img,-1).astype(np.float32))

        return img, footprint, crs
        

    def forward(self, row, row_meta, device='cuda'):

        # Read file
        img, footprint, crs = self._read_image(row)

        # Fragment the sample
        fragments, xys = fragment_fn(img, **self.frag_params, return_indices=True, verbose=False)

        nrows, ncols, c, h, w = fragments.shape
        # Apply the model
        with torch.no_grad():
            embeddings = self.embedder(fragments.reshape(-1,c,h,w).to(device)).view(nrows, ncols, -1)

        df_rows = []

        # Pack rows for geoparquet
        for r_idx in range(nrows):
            for c_idx in range(ncols):
                embedding = embeddings[r_idx, c_idx].cpu().numpy()
                # spatial features per fragment
                x_offset,y_offset=xys[r_idx,c_idx].int().tolist()
                pixel_bbox = [x_offset, y_offset, x_offset + h,y_offset + w] # in pixels
                utm_footprint = crop_footprint(footprint, *img.shape[:2], pixel_bbox)
                # main footprint is in WGS84 (needs to be consistent across parquet)
                transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
                geometry = transform(transformer.transform, utm_footprint) # WGS84
                centre_lon, centre_lat = geometry.centroid.coords[0]
                
                row_dict = {
                    'unique_id' : self.calculate_checksum(geometry, row_meta.timestamp.item(), row_meta.product_id.item(), embedding),
                    'embedding' : embedding,
                    'timestamp' : row_meta.timestamp.item(),
                    'product_id' : row_meta.product_id.item(),
                    'grid_cell' : row_meta.grid_cell.item(),
                    'grid_row_u' : row_meta.grid_row_u.item(),
                    'grid_col_r' : row_meta.grid_col_r.item(),
                    'geometry' : geometry,
                    'centre_lat' : centre_lat,
                    'centre_lon' : centre_lon,
                    'utm_footprint' : utm_footprint.wkt,
                    'utm_crs' : crs.to_string(),
                    'pixel_bbox' : pixel_bbox,
                }
                df_rows.append(row_dict)

        return gpd.GeoDataFrame(df_rows).astype(self.column_types)