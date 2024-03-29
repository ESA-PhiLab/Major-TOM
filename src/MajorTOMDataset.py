import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
from PIL import Image
import torchvision.transforms as transforms

class MajorTOM(Dataset):
    """MajorTOM Dataset (https://huggingface.co/Major-TOM)

    Args:
        df ((geo)pandas.DataFrame): Metadata dataframe
        local_dir (string): Root directory of the local dataset version
        tif_bands (list): A list of tif file names to be read
        png_bands (list): A list of png file names to be read
        
    """
    
    def __init__(self,
                 df,
                 local_dir = None,
                 tif_bands=['B04','B03','B02'],
                 png_bands=['thumbnail'],
                 tif_transforms=[transforms.ToTensor()],
                 png_transforms=[transforms.ToTensor()]
                ):
        super().__init__()
        self.df = df
        self.local_dir = Path(local_dir) if isinstance(local_dir,str) else local_dir
        self.tif_bands = tif_bands if not isinstance(tif_bands,str) else [tif_bands]
        self.png_bands = png_bands if not isinstance(png_bands,str) else [png_bands]
        self.tif_transforms = transforms.Compose(tif_transforms) if tif_transforms is not None else None
        self.png_transforms = transforms.Compose(png_transforms) if png_transforms is not None else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        meta = self.df.iloc[idx]

        product_id = meta.product_id
        grid_cell = meta.grid_cell
        row = grid_cell.split('_')[0]
    
        path = self.local_dir / Path("{}/{}/{}".format(row, grid_cell, product_id))
        out_dict = {'meta' : meta}
        
        for band in self.tif_bands:
            with rio.open(path / '{}.tif'.format(band)) as f:
                out = f.read()
            if self.tif_transforms is not None:
                out = self.tif_transforms(out)
            out_dict[band] = out


        for band in self.png_bands:
            out = Image.open(path / '{}.png'.format(band))
            if self.png_transforms is not None:
                out = self.png_transforms(out)
            out_dict[band] = out

        return out_dict
