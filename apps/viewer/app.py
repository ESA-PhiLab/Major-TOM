import sys
import os
import leafmap
from helpers.grid import *
from helpers.functional import *
from leafmap.toolbar import change_basemap
from IPython.display import display
import ipywidgets
import solara

instructions_top = '''
![image/png](https://raw.githubusercontent.com/ESA-PhiLab/Major-TOM/refs/heads/main/Major_Tom_cropped_logo.png)
# Dataset Viewer
This app provides a way of exploring samples present in the MajorTOM-Core dataset.
It contains nearly every piece of Earth captured by ESA Sentinel-2 satellite as well as a large fraction of paired Sentinel-1 data from a similar time period.
### Instructions
There are **two ways to query the dataset**:

- To find a sample, navigate on the map to a place of interest. Click `Find Sample` to find a dataset sample that contains the central pixel of your current view.
- or you can enter a text prompt (encoded with [SigLIP](https://huggingface.co/datasets/Major-TOM/Core-S2RGB-SigLIP)) and click `Search with Text`
'''

instructions_bottom = '''
<details><summary><strong>🏝 Couldn't find a sample? See this figure of global coverage:</strong></summary>
    <img src='https://cdn-uploads.huggingface.co/production/uploads/6304c06eeb6d777a838eab63/2KTarfsM0a1dNYEbXriUH.png' />
</details>
'''

image_data = solara.reactive(Image.new('RGB',(1068,1068)))
center = solara.reactive((20, 0))
gridcell = solara.reactive('')
timestamp = solara.reactive('')
zoom= solara.reactive(4)
source = solara.reactive('Sentinel-2 L2A')
search_prompt = solara.reactive("")

@solara.component
def Page():
    with solara.Column(style='background-image: linear-gradient(to right, #2f3192, #0171bd, #3ab64b, #f25a24);'):

        with solara.Card(margin=10):
            solara.Markdown(instructions_top)
            solara.Button(label="Website",
                  icon_name="mdi-map-legend",
                  attributes={"href": 'https://www.huggingface.co/Major-TOM', "target": "_blank"},
                  text=True,
                  outlined=True)
    
            solara.Button(label="arXiv Paper",
                          icon_name="mdi-script-text",
                          attributes={"href": 'https://www.arxiv.org/abs/2402.12095', "target": "_blank"},
                          text=True,
                          outlined=True)
            solara.Markdown(instructions_bottom)
        
        def update_image():
            ret = map_to_image(m, return_centre=True, return_gridcell=True, return_timestamp=True, source=source.value)
            if ret is not None:
                image_data.value, center.value, gridcell.value, timestamp.value = ret
                zoom.value=12
            else:
                image_data.value = Image.new('RGB',(1068,1068))
                center.value = (20,0)
                zoom.value = 4

        def update_image_with_text():
            ret = text_to_image(search_prompt.value, return_centre=True, return_gridcell=True, return_timestamp=True, source=source.value)
            if ret is not None:
                image_data.value, center.value, gridcell.value, timestamp.value = ret
                zoom.value=12
            else:
                image_data.value = Image.new('RGB',(1068,1068))
                center.value = (20,0)
                zoom.value = 4

        def update_source(val):
            source.value = val
            update_image()
        
        with solara.Card(margin=10):
            with solara.ColumnsResponsive(default=12, small=12, medium=6, large=6, xlarge=6):
                with solara.Column(align='center'):
                    m = leafmap.Map(
                        height=560,
                        width=560,
                        layout=ipywidgets.Layout(max_width='60vw', max_height='80vh'),
                        zoom=zoom.value,
                        center=center.value,
                        on_center=center.set,
                        draw_control=False,
                        measure_control=False,
                        fullscreen_control=False,
                        toolbar_control=False,
                        attribution_control=True,
                    )
                    display(m)
                    button = solara.Button("Find Sample at Map Center", on_click=update_image, height='80px')                    

                    with solara.Row():
                        solara.InputText("(Optional) Text Prompt:", value=search_prompt)
                        text_button = solara.Button("Search with Text (SigLIP)", on_click=update_image_with_text, height='50px') 
                    
                    solara.Select(label="Data Source", value=source, values=['Sentinel-2 L2A','Sentinel-2 L1C', 'Sentinel-1 RTC', 'COP-DEM','AlphaEarth Embeddings', 'VIIRS Nighttime Light'], on_value=update_source)
                with solara.Column(align='center'):
                    output = solara.Image(image_data.value)
                    solara.Markdown('''        
                    |                     |      |
                    |--------------------:|:----:|
                    | Latitude, Longitude | {:.3f}, {:.3f} |
                    |       MajorTOM Grid | {} |
                    '''.format(center.value[0], center.value[1], gridcell.value))