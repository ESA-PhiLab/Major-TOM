import numpy as np
import geopandas as gpd
import hashlib
from rasterio.io import MemoryFile

from .grid_cell_fragment import *
from .inputs import Image, from_core_row, from_sample, from_taco
from .models import *

class MajorTOM_Embedder(torch.nn.Module):
    """
    MajorTOM Embedder class that applies a model to geospatial image fragments, 
    computes embeddings, and returns metadata for each fragment.

    This class is designed to work with raster data, where the image is fragmented 
    into smaller tiles, and embeddings are computed for each tile using the provided 
    embedder model. The output is a GeoDataFrame containing spatial metadata and 
    the corresponding embeddings for each tile.

    Attributes:
        embedder: A model that generates embeddings for image fragments.
        frag_params: Dictionary containing fragmentation parameters such as the 
                      target overlap and border shift.
        column_types: Dictionary specifying data types for the output GeoDataFrame columns.
    """
    
    def __init__(self, embedder, target_overlap=0.1, border_shift=True):
        """
        Initializes the MajorTOM Embedder with the given parameters.

        Args:
            embedder (torch.nn.Module): A model that generates embeddings for image fragments.
            target_overlap (float): The target overlap between image fragments. Default is 0.1.
            border_shift (bool): Whether to shift the borders of fragments to avoid edge artifacts. Default is True.
        """
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
        """
        Returns the set of input bands in the correct order.

        Returns:
            list: List of input bands used by the embedder.
        """
        return self.embedder.bands

    def size(self):
        """
        Returns the input image size.

        Returns:
            tuple: Tuple representing the image size (height, width).
        """
        return self.embedder.size

    def calculate_checksum(self, geometry, timestamp, product_id, embedding):
        """
        Calculates a checksum for the given geometry, timestamp, product ID, and embedding.

        Args:
            geometry (shapely.geometry): The geometry object representing the fragment's footprint.
            timestamp (str): Timestamp of the data.
            product_id (str): Product identifier.
            embedding (np.ndarray): The embedding of the image fragment.

        Returns:
            str: A SHA256 checksum of the concatenated input parameters.

        The embedding enters as numpy's string of the float64 array, which numpy abbreviates for more than 1,000
        values (only the first and last 3 are written). The published Core embeddings reproduce this way
        (200 of 200 Core-S2RGB-SigLIP rows checked, 2026-11-06): recompute with the stored embedding cast to float64.
        """
        embedding = np.asarray(embedding, dtype=np.float64)
        combined = f"{geometry}_{timestamp}_{product_id}_{embedding}"
        checksum = hashlib.sha256(combined.encode()).hexdigest()
        return checksum

    def forward(self, row, row_meta, device='cuda'):
        """
        Embeds one Core sample: a row group of a Core image Parquet file and its metadata row.

        Args:
            row (pyarrow.Table): one row group of a Core image Parquet file (bands as GeoTIFF bytes).
            row_meta (pandas.DataFrame): the sample's row of the dataset's metadata.parquet (one row).
            device (str): 'cuda' or 'cpu'.

        Returns:
            geopandas.GeoDataFrame: one row per fragment (see `embed`).
        """
        return self.embed(from_core_row(row, row_meta, self.embedder.bands), device=device)

    def embed_sample(self, sample, device='cuda'):
        """Embeds a majortom.build sample (path to its .npz; the .json sidecar beside it)."""
        return self.embed(from_sample(sample, self.embedder.bands), device=device)

    def embed_taco(self, record, device='cuda'):
        """Embeds one record (catalogue row as a dict) of a TACO dataset written by majortom.release."""
        return self.embed(from_taco(dict(record), self.embedder.bands), device=device)

    def embed(self, image, device='cuda'):
        """
        Fragments an Image, embeds each fragment and returns one row per fragment.

        Columns: unique_id (SHA-256 of geometry WKT, timestamp, product_id and the embedding's float64 string;
        see `calculate_checksum`), embedding, timestamp, product_id, grid_cell, grid_row_u, grid_col_r,
        geometry (WGS84), centre_lat, centre_lon, utm_footprint (WKT in utm_crs), utm_crs,
        pixel_bbox ([x0, y0, x1, y1] in the image's pixels).

        Args:
            image (Image): from majortom.embedder.inputs (from_core_row, from_sample, from_taco).
            device (str): 'cuda' or 'cpu'.
        """
        img = torch.from_numpy(image.data)
        footprint = box(*image.bounds)
        crs = CRS.from_user_input(image.crs)
        row_u, col_r = (int(v[:-1]) * (1 if v[-1] in 'UR' else -1) for v in image.grid_cell.split('_'))

        fragments, xys = fragment_fn(img, **self.frag_params, return_indices=True, verbose=False)
        nrows, ncols, c, h, w = fragments.shape
        with torch.no_grad():
            embeddings = self.embedder(fragments.reshape(-1, c, h, w).to(device)).view(nrows, ncols, -1)

        to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        df_rows = []
        for r_idx in range(nrows):
            for c_idx in range(ncols):
                embedding = embeddings[r_idx, c_idx].cpu().numpy()
                x_offset, y_offset = xys[r_idx, c_idx].int().tolist()
                pixel_bbox = [x_offset, y_offset, x_offset + h, y_offset + w]                  # in pixels
                utm_footprint = crop_footprint(footprint, *img.shape[:2], pixel_bbox)
                geometry = transform(to_wgs84.transform, utm_footprint)                       # WGS84
                centre_lon, centre_lat = geometry.centroid.coords[0]
                df_rows.append({
                    'unique_id': self.calculate_checksum(geometry, image.timestamp, image.product_id, embedding),
                    'embedding': embedding,
                    'timestamp': image.timestamp,
                    'product_id': image.product_id,
                    'grid_cell': image.grid_cell,
                    'grid_row_u': row_u,
                    'grid_col_r': col_r,
                    'geometry': geometry,
                    'centre_lat': centre_lat,
                    'centre_lon': centre_lon,
                    'utm_footprint': utm_footprint.wkt,
                    'utm_crs': crs.to_string(),
                    'pixel_bbox': pixel_bbox,
                })
        return gpd.GeoDataFrame(df_rows, geometry='geometry', crs='EPSG:4326').astype(self.column_types)
