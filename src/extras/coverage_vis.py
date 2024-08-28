import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import PIL

def get_mask(df):
    """
        Take a Major TOM dataframe and create a mask corresponding to available cells
    """
    
    mask = np.zeros((2004,4008), dtype=np.uint8)
    row_offset = -1002
    col_offset = -2004
    
    nodata = df['nodata'].values > 0.5
    
    yy = mask.shape[0] - (np.array(df['grid_row_u']) - row_offset) - 1
    xx = np.array(df['grid_col_r']) - col_offset
    
    yy = yy[~nodata]
    xx = xx[~nodata]
    
    mask[yy, xx] = 255

    return PIL.Image.fromarray(mask)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def light_basemap():
    """
        Bright coloured contours
    """
    
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(48,24), dpi=167)
    
        m = Basemap(projection='sinu', lat_0=0, lon_0=0, resolution='l', ax=ax)
        m.fillcontinents(color="#9eba9b", lake_color='#CCDDFF')
        m.drawmapboundary(fill_color="#CCDDFF")
        m.drawcountries(color="#666666", linewidth=1)
        m.drawcoastlines(color="#666666", linewidth=1)
        
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        return fig2img(fig)

def dark_basemap():
    """
        Dark contours
    """
    
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(48,24), dpi=167)
        
        m = Basemap(projection='sinu', lat_0=0, lon_0=0, resolution='l', ax=ax)
        m.fillcontinents(color="#242424", lake_color='#242424')
        m.drawmapboundary(fill_color="#242424")
        m.drawcountries(color="#000000", linewidth=1)
        m.drawcoastlines(color="#000000", linewidth=1)
        
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        return fig2img(fig)

def get_coveragemap(input):
    """
        Creates a complete coloured Major TOM coverage figure in the same style as in the official documentation
    """

    # compute mask if df is provided
    if isinstance(input, pd.DataFrame):
        mask = get_mask(input)
    else:
        mask = input
    
    basemap = light_basemap()
    basemap_d = dark_basemap()
        
    outside_earth = np.array(basemap.convert('RGBA'))[:, :, 0] == 255
    outside_earth = PIL.Image.fromarray(outside_earth)
    
    mask = mask.resize(basemap.size, PIL.Image.NEAREST)
    
    basemap.putalpha(mask)
    
    # Mask outside of earth
    basemap.paste(outside_earth, (0,0), outside_earth)
    
    basemap_d.paste(basemap, (0,0), basemap)

    return basemap_d

if __name__ == '__main__':
    DATASET_NAME = 'Major-TOM/Core-S2L2A'
    meta_path = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet'.format(DATASET_NAME)
    
    df = pd.read_parquet(meta_path)

    # This is how you make a coverage figure!
    coverage_img = get_coveragemap(df)

    coverage_img.save('coverage-example.png', format='PNG')
