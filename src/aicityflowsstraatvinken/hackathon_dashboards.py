from sklearn.pipeline import Pipeline
import streamlit as st
from streamlit_folium import folium_static
import folium
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import yaml
import pickle
import altair as alt
import shap
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

import streamlit.components.v1 as components

class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.names]


FILE_PATTERN = "data/processed/*infer_output*.csv"
INFER_CONFIG_PATH = "src/aicityflowsstraatvinken/infer_config_2022_3.yaml"


config = yaml.load(stream=open(INFER_CONFIG_PATH, 'r'), Loader=yaml.FullLoader)


#file = st.radio("input file", glob(FILE_PATTERN))
file='data/processed/20220404-infer_output_quentin_small.csv'
#tiles = st.radio("tiles", ["CartoDb positron", "Stamen Toner", "Stamen Terrain", "Stamen Watercolor", "OpenStreetMap"])
tiles = "CartoDb positron"
input_df = pd.read_csv(file)
input_df["geometry"] = input_df["geometry"].apply(lambda x: wkt.loads(x))
#print(input_df.modal_dist)
input_df["street_segment"] = input_df["street_segment"].apply(lambda x: wkt.loads(x))
#print(input_df.geometry)
input_gpd = gpd.GeoDataFrame(input_df, geometry=input_df.street_segment)
input_gpd["modal_split"] = input_gpd[["bike","walk"]].sum(axis=1) / input_gpd[["truck","bus","van","car","bike","walk"]].sum(axis=1)
input_gpd["modal_split"] = input_gpd["modal_split"]*100
input_gpd["pae"] = input_gpd["car"] + input_gpd["van"] +  2 * input_gpd["truck"] + 2 * input_gpd["bus"]
show_cols = [col for col in ["walk", "truck","bus","van","car","bike", "modal_split", "pae"] if col in input_gpd.columns]
input_gpd[show_cols] = input_gpd[show_cols].astype(int)

one_hot_cols = config["columns"]["one_hot_cols"]
one_hot_categories = config["columns"]["one_hot_categories"]

for ix, col in enumerate(one_hot_cols):
   def clean_unknown(item, known=one_hot_categories[ix]):
       return item if item in known else known[0]
   input_gpd[col] = input_gpd[col].apply(lambda x: clean_unknown(x))

show=st.sidebar.radio("show", show_cols)

f"### Straatvinken ML - {show}"
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

cmap = st.sidebar.radio("colormap", ["RdYlBu_r", "YlOrRd", "Greens", "RdYlGn", "RdYlBu"])
binning = st.sidebar.radio("bins", ['NaturalBreaks', 'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled', 'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced', 'JenksCaspallSampled', 'MaxP', 'MaximumBreaks', 'Quantiles', 'Percentiles', 'StdMean'])
#print(input_gpd[["geometry", "street_segment"]])
max_percentile = st.sidebar.slider("max percentile", 80, 100, 100, help="""
if you have large outliers and nearly all observations are in a single bin, 
you can set the maximum percentile to a value below 100 to change the maximum value for binning""")
vmax =  np.percentile(input_gpd[show].values, max_percentile)
explain=st.sidebar.number_input("explain (WS_OIDN)", value=409989)



#calculate center
minx, miny, maxx, maxy = input_gpd.geometry.total_bounds
centerx, centery=(maxx-(maxx-minx)/2), (maxy-(maxy-miny)/2)
#print(centerx, centery)

######
###### MAP VISUALIZATION
######

m= folium.Map(location=[centery, centerx], zoom_start=15, tiles=tiles)
m.add_child(folium.plugins.Fullscreen(position="topright"))
show_cols.append("geometry")
if "WS_OIDN" in input_gpd.columns:
    show_cols.append("WS_OIDN")
m = input_gpd[
    show_cols
].explore(column=show,
    m=m,
    vmin=0,
    vmax=vmax,
    k=10, 
    scheme=binning, 
    cmap=cmap,
    #marker_type="marker"
)

#col1, col2 = st.columns([2, 1])

# call to render Folium map in Streamlit
#with col1:
folium_static(m, width=800, height=600)

#with col2:
    #####
    ##### FEATURE IMPORTANCE
    #####
Y_s = config["columns"]["Y_s"]
one_hot_cols = config["columns"]["one_hot_cols"]
num_pred_remain_cols = config["columns"]["num_pred_remain_cols"]
num_pred_segm_cols = config["columns"]["num_pred_segm_cols"]
num_pred_minmax_cols = config["columns"]["num_pred_minmax_cols"]
one_hot_categories = config["columns"]["one_hot_categories"]
one_hot_hotter = []

for cat, unts in zip(one_hot_cols, one_hot_categories):
    one_hot_hotter.extend([f"{cat}_{unt}" for unt in unts[1:]])

num_pred_minmax = num_pred_minmax_cols.copy()
num_pred_minmax.extend(num_pred_segm_cols)
cols = one_hot_cols.copy()
cols.extend(num_pred_minmax)
cols.extend(num_pred_remain_cols)

train_cols = one_hot_hotter.copy()
train_cols.extend(num_pred_minmax)
train_cols.extend(num_pred_remain_cols)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if show in config["infer_models"]:
    model_name = config["infer_models"][show]["model"]
    model_path = config["infer_models"][show]["model_path"]
    model = pickle.load(open(model_path, "rb"))
    feat_lim=25
    if model_name == "CatBoostRegressor":
        f"__Feature importance for '{show}' ({model_name}, top {feat_lim})__"
        feat_data = model.named_steps["est"].get_feature_importance()
        source = pd.DataFrame({"column": train_cols, "importance": feat_data})
        source = source.sort_values(by=["importance"], ascending=False).reset_index(drop=True).iloc[:feat_lim]

        bars = alt.Chart(source).mark_bar().encode(
            x='importance:Q',
            y=alt.Y('column:N', sort='-x')
        )

        st.altair_chart((bars).properties(width=400), use_container_width=True)
    elif model_name == "LGBMRegressor":
        f"__Feature importance for '{show}' ({model_name}, top {feat_lim})__"
        source = pd.DataFrame(sorted(zip(model.named_steps["est"].feature_importances_, train_cols)), columns=['importance','column'])
        source = source.sort_values(by=["importance"], ascending=False).reset_index(drop=True).iloc[:feat_lim]

        bars = alt.Chart(source).mark_bar().encode(
            x='importance:Q',
            y=alt.Y('column:N', sort='-x')
        )

        st.altair_chart(bars, use_container_width=True)
    else:
        "No feature importance for this model type!"
    
    if explain not in input_gpd.WS_OIDN.values:
            f"No explanation for '{explain}'"
    else:

        transformer = pickle.load(open("data/model/20220330_transformer.pkl", "rb"))
        
        explainer = shap.TreeExplainer(model.named_steps["est"])
        sample_data = input_gpd[input_gpd.WS_OIDN == explain][cols]
        # print(sample_data.shape)    
        trans_data = transformer.transform(sample_data)
        # print(trans_data)
        # print(trans_data.shape)
        # print(train_cols)
        # print(len(train_cols))
        print(model.named_steps["est"].predict(trans_data))

        sample_explain = pd.DataFrame(trans_data, columns=train_cols)
        shap_values = explainer.shap_values(sample_explain)
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        f"__Explanation for WS_OIDN '{explain}'__"
        st_shap(shap.force_plot(
            explainer.expected_value, 
            shap_values, 
            sample_explain,
            #matplotlib=True,
            #show=False,
            figsize=(16,5))
        )
else:
    f"No model behind '{show}' (calculated value)"

    