import sys
import os
import leafmap
from helpers.grid import *
from helpers.functional import *
from leafmap.toolbar import change_basemap
from IPython.display import display
import ipywidgets
import ipyleaflet
import solara

logo = 'https://raw.githubusercontent.com/ESA-PhiLab/Major-TOM/refs/heads/main/Major_Tom_cropped_logo.png'

# collapsed under the header: what the app is, how to query, global coverage
instructions = '''
This app provides a way of exploring samples present in the MajorTOM-Core dataset.
It contains nearly every piece of Earth captured by ESA Sentinel-2 satellite as well as a large fraction of paired Sentinel-1 data from a similar time period.

There are **two ways to query the dataset**:

- To find a sample, navigate on the map to a place of interest. Click `Find Sample` to find a dataset sample that contains the central pixel of your current view.
- or you can enter a text prompt (encoded with [SigLIP](https://huggingface.co/datasets/Major-TOM/Core-S2RGB-SigLIP)) and click `Search with Text`

Couldn't find a sample? Global coverage:

<img src='https://cdn-uploads.huggingface.co/production/uploads/6304c06eeb6d777a838eab63/2KTarfsM0a1dNYEbXriUH.png' width='100%' />
'''

# Inter for all text (icons keep their font: theirs is set on ::before), plain-case buttons, even spacing
css = '''
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.v-application, .v-application *:not(.v-icon):not(.mdi) { font-family: 'Inter', system-ui, sans-serif !important; }
.v-application { color: #1f2328; font-size: 15px; }
.v-btn { text-transform: none !important; letter-spacing: 0 !important; font-weight: 500 !important; }
.v-card { border-radius: 12px !important; }
.mt-page { padding: 24px 24px 56px; gap: 24px !important; }
.mt-panel { width: 100%; max-width: 640px; margin: 0 auto; gap: 12px; }
.mt-search { width: 100%; flex-wrap: nowrap; align-items: center; gap: 8px; }
.mt-panel table { width: 100%; border-collapse: collapse; font-size: 14px; }
.mt-panel th { display: none; }
.mt-panel td { padding: 6px 4px; border-bottom: 1px solid #eaeef2; }
.mt-panel td:first-child { color: #59636e; width: 45%; }
.mt-panel img { border-radius: 8px; }
.mt-panel tr { background: none !important; }
'''

BLUE = '#0171bd'                              # the Major TOM mark's blue
PLACEHOLDER = (241, 243, 245)                # light grey: no sample shown yet
SOURCES = ['Sentinel-2 L2A', 'Sentinel-2 L1C', 'Sentinel-1 RTC', 'COP-DEM', 'AlphaEarth Embeddings', 'VIIRS Nighttime Light']
image_data = solara.reactive(Image.new('RGB', (1068, 1068), PLACEHOLDER))
center = solara.reactive((20, 0))
gridcell = solara.reactive('')
timestamp = solara.reactive('')
zoom= solara.reactive(4)
source = solara.reactive('Sentinel-2 L2A')
search_prompt = solara.reactive("")

@solara.component
def SearchBox(on_search):
    """The text prompt and its button. Its own component: typing re-renders only this row, not the page and its map."""
    with solara.Row(classes=["mt-search"]):
        prompt = solara.v.TextField(v_model=search_prompt.value, on_v_model=search_prompt.set,
                                    label="Search by description (SigLIP)",
                                    placeholder="e.g. harbour with ships", variant="outlined", density="compact",
                                    hide_details=True, clearable=True,
                                    prepend_inner_icon="mdi-text-search")
        solara.v.use_event(prompt, "keyup.enter", lambda *args: on_search())
        solara.Button("Search", icon_name="mdi-magnify", on_click=on_search, color=BLUE, outlined=True)


@solara.component
def Page():
    solara.Style(css)
    with solara.Column(classes=["mt-page"], style='background-image: linear-gradient(to right, #2f3192, #0171bd, #3ab64b, #f25a24);'):

        with solara.Card(margin=0):
            # header: logo, title, links in one row; the instructions folded below it
            with solara.Row(style={"align-items": "center", "flex-wrap": "wrap", "gap": "16px"}):
                solara.HTML(tag="img", attributes={"src": logo, "alt": "Major TOM"}, style="height: 48px")
                solara.Text("Dataset Viewer", style={"font-size": "24px", "font-weight": "500"})
                solara.v.Spacer()
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
            solara.Details(summary="Instructions and coverage", children=[solara.Markdown(instructions)])
        
        def update_image():
            ret = map_to_image(m, return_centre=True, return_gridcell=True, return_timestamp=True, source=source.value)
            if ret is not None:
                image_data.value, center.value, gridcell.value, timestamp.value = ret
                zoom.value=12
            else:
                image_data.value = Image.new('RGB', (1068, 1068), PLACEHOLDER)
                center.value = (20,0)
                zoom.value = 4

        def update_image_with_text():
            ret = text_to_image(search_prompt.value, return_centre=True, return_gridcell=True, return_timestamp=True, source=source.value)
            if ret is not None:
                image_data.value, center.value, gridcell.value, timestamp.value = ret
                zoom.value=12
            else:
                image_data.value = Image.new('RGB', (1068, 1068), PLACEHOLDER)
                center.value = (20,0)
                zoom.value = 4

        def update_source(val):
            source.value = val
            update_image()
        
        with solara.Card(margin=0):
            with solara.ColumnsResponsive(default=12, small=12, medium=6, large=6, xlarge=6):
                with solara.Column(classes=["mt-panel"]):
                    solara.v.Select(v_model=source.value, on_v_model=update_source, items=SOURCES, label="Data source",
                                    variant="outlined", density="compact", hide_details=True)
                    m = leafmap.Map(
                        height=560,
                        layout=ipywidgets.Layout(width='100%', max_height='80vh'),
                        zoom=zoom.value,
                        center=center.value,
                        on_center=center.set,
                        draw_control=False,
                        measure_control=False,
                        fullscreen_control=False,
                        toolbar_control=False,
                        attribution_control=True,
                    )
                    for extra in [c for c in m.controls if isinstance(c, ipyleaflet.AttributionControl)][1:]:
                        m.remove(extra)                       # leafmap adds a second one: the attribution showed twice
                    display(m)
                    solara.Button("Find sample at map centre", icon_name="mdi-crosshairs-gps", on_click=update_image,
                                  color=BLUE, block=True)
                    SearchBox(update_image_with_text)
                with solara.Column(classes=["mt-panel"]):
                    solara.Image(image_data.value, width="100%")
                    solara.Markdown(f'''
| | |
|:--|:--|
| Grid cell | {gridcell.value or "–"} |
| Latitude, longitude | {center.value[0]:.3f}, {center.value[1]:.3f} |
| Acquired | {timestamp.value or "–"} |
| Source | {source.value} |
''', style={"width": "100%"})
