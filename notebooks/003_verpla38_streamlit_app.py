import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle
import brewer2mpl

DATA_URL = (
    "data/processed/20220222_streetview_coordinates_w_labels.pkl"
)

CLASSES_URL = (
    "data/external/CSAILVision/object150_info.csv"
)

st.title("Straatvinken - Google Streetview landscape segmentation demo")
st.markdown(
"""
This is a geo visualization of the streetview image segmentation data for the region of Flanders (currently only Antwerp)
""")

data = pickle.load(open(DATA_URL, "rb"))
#print(data.columns)
#data["lat"] = data.location.apply(lambda x: x["lat"])
#data["lon"] = data.location.apply(lambda x: x["lng"])

style_selectbox = st.sidebar.selectbox(
    'Map style',
    (
        'light',
        'dark',
        'road',
        'satellite'
    )
)

clrscale_selectbox = st.sidebar.selectbox(
    'Color scale',
    (
        'YlGnBu - sequential - 9' ,
        'YlOrRd - sequential - 9',
        'Blues - sequential - 9' ,
        'BuGn - sequential - 9'   ,
        'BuPu - sequential - 9'   ,
        'GnBu - sequential - 9'   ,
        'Greens - sequential - 9' ,
        'Greys - sequential - 9'  ,
        'OrRd - sequential - 9'   ,
        'Oranges - sequential - 9',
        'PuBu - sequential - 9'   ,
        'PuBuGn - sequential - 9' ,
        'PuRd - sequential - 9'   ,
        'Purples - sequential - 9',
        'RdPu - sequential - 9'   ,
        'Reds - sequential - 9'   ,
        'YlGn - sequential - 9'   ,
        'YlOrBr - sequential - 9' ,
        'BrBG - diverging - 11',
        'PRGn - diverging - 11',
        'PiYG - diverging - 11',
        'PuOr - diverging - 11',
        'RdBu - diverging - 11',
        'RdGy - diverging - 11',
        'RdYlBu - diverging - 11',
        'RdYlGn - diverging - 11',
        'Spectral - diverging - 11',
    )
)

segment_selectbox = st.sidebar.selectbox(
    'Segment',
    (
        'sky',
        'road',
        'building',
        'tree',
        'grass',
        'sidewalk',
        'plant',
        'car',
        'earth',
        'wall',
        'fence',
        'field',
        'floor',
        'ceiling',
        'house',
        'path',
        'water',
        'truck',
        'signboard',
        'van'
    )
)

COLOR_SCALE = brewer2mpl.get_map(clrscale_selectbox.split(" - ")[0], clrscale_selectbox.split(" - ")[1], int(clrscale_selectbox.split(" - ")[2])).colors


img_width = 300
#data["value"] = data[segment_selectbox]
#data = data[data.value.notnull()]
st.subheader(f"Segment: {segment_selectbox}")
midpoint = (np.average(data["lat"]), np.average(data["long"]))
deck = pdk.Deck(
    map_style=f"mapbox://styles/mapbox/{style_selectbox}-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 45,
    },
    layers=[
        pdk.Layer(
            "ColumnLayer",
            data=data,
            get_position=["long", "lat"],
            get_elevation = segment_selectbox,
            elevation_scale=0,
            opacity=1.0,
            pickable=True,
            radius=50,
            get_fill_color=[0, 0, 0, 30],
            auto_highlight=True,
        ),
        pdk.Layer(
            "HeatmapLayer",
            data,
            opacity=0.2,
            get_position=["long", "lat"],
            aggregation='"MEAN"',
            color_range=COLOR_SCALE,
            radius=100,
            get_weight="%s / %s" % (segment_selectbox, data[segment_selectbox].max()),
            )
    ],
    height=500,
    width=800,
    tooltip = {
        "html": """
        <b>{%s}%%</b> is <b>%s</b><br /> 
        <table>
            <tr>
                <td><img src='http://localhost:8000/{pano_id}_0_N_0.jpg' width='%spx' /></td>
                <td><img src='http://localhost:8000/{pano_id}_1_E_90.jpg' width='%spx' /></td>
            </tr>
            <tr>
                <td><img src='http://localhost:8000/{pano_id}_2_S_180.jpg' width='%spx' /></td>
                <td><img src='http://localhost:8000/{pano_id}_3_W_270.jpg' width='%spx' /></td>
            </tr>
        </table>
        <br />
        location: {lat, long}
        """ % (segment_selectbox, segment_selectbox, img_width, img_width, img_width, img_width),
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"}
    }
)

st.write(deck)