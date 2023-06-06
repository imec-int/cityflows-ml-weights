import yaml
import pickle
import time
import pandas as pd
import geopandas as gpd
import sys
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(".")), "ai-cityflows-straatvinken", "src", "aicityflowsstraatvinken"))
print(sys.path)
from utils import *

BASE_DIR="data/raw/"
IDENTIFIER="coordinates_quentin_large"
SUFFIX=0
print("suffix", SUFFIX)
if SUFFIX != "":
    COORDINATE_PATH = f"{BASE_DIR}infer_{IDENTIFIER}_{SUFFIX}.csv"
else:
    COORDINATE_PATH = f"{BASE_DIR}infer_{IDENTIFIER}.csv"

ITERATION="2022_3"
infer_day = "20220525"
#infer_day = date.today().isoformat().replace("-","")
INFER_OUTPUT_PATH = f"data/processed/{infer_day}-infer_{IDENTIFIER}_output_{SUFFIX}.csv"
INFER_NO_OUTPUT_PATH = INFER_OUTPUT_PATH.replace("output","no_output")#.replace("csv","pkl")
INFER_CONFIG_PATH = f"src/aicityflowsstraatvinken/infer_config_{ITERATION}.yaml"

infer_filetype = INFER_OUTPUT_PATH.split(".")[-1]
#load no output data

if os.path.exists(INFER_NO_OUTPUT_PATH):
    if infer_filetype == 'csv':
        with open(INFER_NO_OUTPUT_PATH, "r") as noout_file:
            infer_df = pd.read_csv(noout_file)
            infer_gpd = gpd.GeoDataFrame(infer_df, geometry=gpd.points_from_xy(infer_df.long, infer_df.lat), crs=4326)
    elif infer_filetype == "pkl":
        with open(INFER_NO_OUTPUT_PATH, "rb") as noout_file:
            infer_gpd = pickle.load(noout_file)
    print(f"skipped dataset creation because '{INFER_NO_OUTPUT_PATH}' exists")

#load inference config
config = yaml.load(stream=open(INFER_CONFIG_PATH, 'r'), Loader=yaml.FullLoader)

#target columns
Y_s = config["columns"]["Y_s"]

#predictor columns
one_hot_cols = config["columns"]["one_hot_cols"]
one_hot_categories = config["columns"]["one_hot_categories"]
# TODO: this should not be necessary (all possible categories should be included)
for ix, col in enumerate(one_hot_cols):
    def clean_unknown(item, known=one_hot_categories[ix]):
        return item if item in known else known[0]
    infer_gpd[col] = infer_gpd[col].apply(lambda x: clean_unknown(x))
    #print(col, pd.unique(infer_gpd[col]))
num_pred_remain_cols = config["columns"]["num_pred_remain_cols"]
num_pred_segm_cols = config["columns"]["num_pred_segm_cols"]
num_pred_minmax_cols = config["columns"]["num_pred_minmax_cols"]
num_pred_minmax = num_pred_minmax_cols.copy()
num_pred_minmax.extend(num_pred_segm_cols)
cols = one_hot_cols.copy()
cols.extend(num_pred_minmax)
cols.extend(num_pred_remain_cols)

#inference on gathered data
for y in Y_s:
    start_y=time.time()

    model_name = config["infer_models"][y]["model"]
    model_path = config["infer_models"][y]["model_path"]

    if model_path == "":
        print(f"No model for modality {y}, skipping")
        infer_gpd[y] = ["no_model"] * infer_gpd.shape[0]
        continue
    with open(f"{model_path}", "rb") as model_file:
        print(f"loading model {model_name} ({model_path}) in")
        model = pickle.load(model_file)
        y_infer = model.predict(infer_gpd[cols])
        infer_gpd[y] = y_infer
        print(f" - inference models for target {y} with model {model_name} performed in {time.time()-start_y:.2f}s")

if infer_filetype == "pkl":
    with open(INFER_OUTPUT_PATH, "wb") as infer_file:
        pickle.dump(infer_file)
elif infer_filetype == "csv":
    with open(INFER_OUTPUT_PATH, "w") as infer_file:
        infer_gpd.to_csv(infer_file)

print(f"inference completed in {time.time()-start:.2f}")