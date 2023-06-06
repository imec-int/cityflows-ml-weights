from io import UnsupportedOperation
import warnings
import argparse

from utils import Columns
warnings.filterwarnings("ignore")

import os
import pandas as pd
import geopandas as gpd
import time
from datetime import date, datetime
import yaml
import pickle

from dotenv import load_dotenv
from utils import get_straatvinken_data
from utils import add_bebouwingsdichtheid
from utils import add_populationdensity
from utils import add_numberofcars
from utils import get_wegsegment_data
from utils import add_wegsegment
from utils import get_featurecount_within_distance
from utils import add_wegsegment_attributes
from utils import add_pop_sm
from utils import add_trafficaccidents
from utils import add_streetview_segmentdata
from utils import add_rura_feature
from utils import add_school_distance_data
from utils import handle_missing_data
from utils import add_distance_data
from utils import add_geometric_feature_availability


def infer(args):
    
    BASE_DIR="data/raw/"
    IDENTIFIER=args.identifier
    SUFFIX=args.suffix
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


    print(f"Performing inference on {datetime.now().isoformat(timespec='seconds')} with file {COORDINATE_PATH}")

    infer_filetype = INFER_OUTPUT_PATH.split(".")[-1]
    if infer_filetype not in ["csv", "pkl"]:
        raise UnsupportedOperation(f"Output path is of unsupported type '{infer_filetype}'")

    start_start = time.time()

    if os.path.exists(INFER_NO_OUTPUT_PATH):
        if infer_filetype == 'csv':
            with open(INFER_NO_OUTPUT_PATH, "r") as noout_file:
                infer_df = pd.read_csv(noout_file)
                infer_gpd = gpd.GeoDataFrame(infer_df, geometry=gpd.points_from_xy(infer_df.long, infer_df.lat), crs=4326)
        elif infer_filetype == "pkl":
            with open(INFER_NO_OUTPUT_PATH, "rb") as noout_file:
                infer_gpd = pickle.load(noout_file)
        print(f"skipped dataset creation because '{INFER_NO_OUTPUT_PATH}' exists")
    else:
        # read coordinates for inference
        infer_df = pd.read_csv(COORDINATE_PATH)
        if not "lat" in infer_df.columns or not "long" in infer_df.columns:
            raise ValueError("Both 'lat' and 'long' columns need to be present in the source file!")

        infer_gpd = gpd.GeoDataFrame(infer_df, geometry=gpd.points_from_xy(infer_df.long, infer_df.lat), crs=4326)
        print(f"Loaded coordinate datafile {COORDINATE_PATH} containing {infer_df.shape[0]} coordinates")
        infer_gpd = infer_gpd.reset_index(drop=True)
        # perform all data merging and feature engineering
        print("Starting feature engineering")

        start = time.time()
        bedi_path=f"{BASE_DIR}lu_bebdicht_ha_vlaa_2019_v2/lu_bebdicht_ha_vlaa_2019_v2.tif"
        infer_gpd = add_bebouwingsdichtheid(
            infer_gpd,
            bedi_path
        )
        print(f" - added bedi in {time.time()-start:.2f}s")

        start = time.time()
        statsect_path=f"{BASE_DIR}sh_statbel_statistical_sectors_20210101.shp/sh_statbel_statistical_sectors_20210101.shp"
        pode_path=f"{BASE_DIR}population_density_statbel/OPENDATA_SECTOREN_2021.xlsx"
        infer_gpd = add_populationdensity(
            infer_gpd, 
            pode_path, 
            statsect_path
        )
        print(f" - added pode in {time.time()-start:.2f}s")

        start = time.time()
        ncars_path = f"{BASE_DIR}number_of_cars_statbel/TF_CAR_HH_SECTOR.xlsx"
        infer_gpd = add_numberofcars(infer_gpd, ncars_path)
        print(f" - added ncars in {time.time()-start:.2f}s")

        start = time.time()
        popu_path=f"{BASE_DIR}TF_POPULATION_GRID_3035_20200101.shp/TF_POPULATION_GRID_3035_20200101.shp"
        infer_gpd = add_pop_sm(
            infer_gpd, 
            popu_path=popu_path
        )
        print(f" - added popu in {time.time()-start:.2f}s")

        start = time.time()
        acc_path = f"{BASE_DIR}traffic_accidents_statbel/TF_ACCIDENTS_2020.xlsx"
        infer_gpd = add_trafficaccidents(
            infer_gpd, 
            acc_path=acc_path,
            statsect_path=statsect_path
        )
        print(f" - added acc in {time.time()-start:.2f}s")

        start = time.time()
        wrsegm_path = f"{BASE_DIR}Wegenregister_SHAPE_20211216/Shapefile/Wegsegment.shp"
        wrsegm_gpd = get_wegsegment_data(wrsegm_path)
        print(f" - loaded wegsegment data in {time.time()-start:.2f}s")

        start = time.time()
        dbf_basedir = f"{BASE_DIR}Wegenregister_SHAPE_20211216/Shapefile/"
        wrsegm_gpd = add_wegsegment_attributes(wrsegm_gpd, dbf_basedir)
        print(f" - added wegsegment attributes in {time.time()-start:.2f}s")

        start = time.time()
        infer_gpd = add_wegsegment(infer_gpd, wrsegm_gpd)
        print(f" - added wr segment data in {time.time()-start:.2f}s")

        infer_gpd = infer_gpd.reset_index(drop=True)
        first_start = time.time()
        
        for max_distance in [50, 250, 1000]:
            start=time.time()
            infer_gpd = get_featurecount_within_distance(
                src_gdf=infer_gpd,
                featuredb_gdf=wrsegm_gpd,
                feature_col="wegcat",
                max_distance=max_distance,
                result_type="feature_columns",
                column_suffix=f"_{max_distance}",
                total_buckets_prefix = ("H", "P", "S", "L"),
                feature_filters = ("-8", "-9"),
                verbose=False
            )
            print(f" - wegcat features within {max_distance} done in {time.time()-start:.2f}s")
            
            start=time.time()
            infer_gpd = get_featurecount_within_distance(
                src_gdf=infer_gpd,
                featuredb_gdf=wrsegm_gpd,
                feature_col="morf",
                max_distance=max_distance,
                result_type="feature_columns",
                column_suffix=f"_{max_distance}",
                feature_filters = ("-8", "-9"),
                verbose=False
            )
            print(f" - morf features within {max_distance} done in {time.time()-start:.2f}s")
        print(f" - added all wr vicinity data in {time.time()-first_start:.2f}s")

        start=time.time()
        school_path = f"{BASE_DIR}Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service_Shapefile/Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service/Shapefile/POI_Onderwijs.shp"
        infer_gpd = add_school_distance_data(infer_gpd, school_path, neighbors=3)
        print(f" - added school distance data in {time.time()-start:.2f}s")

        ### RURA DATA
        start=time.time()
        feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/kernclusters_kernen2019_v2.shp"
        infer_gpd = add_rura_feature(infer_gpd, feature_path, feature_name="kc_kern", buffer_radius=250, geom_type="polygon")
        print(f" - added rura kc_kern data in {time.time()-start:.2f}s")

        start=time.time()
        feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/kernclusters_linten2019_v2.shp"
        infer_gpd = add_rura_feature(infer_gpd, feature_path, feature_name="kc_lint", buffer_radius=250, geom_type="linestring")
        print(f" - added rura kc_lint data in {time.time()-start:.2f}s")
        
        start=time.time()
        feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/kernen2019_v2.shp"
        infer_gpd = add_rura_feature(infer_gpd, feature_path, feature_name="kern", buffer_radius=250, geom_type="polygon")
        print(f" - added rura kern data in {time.time()-start:.2f}s")
        
        start=time.time()
        feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/linten2019_v2.shp"
        infer_gpd = add_rura_feature(infer_gpd, feature_path, feature_name="lint", buffer_radius=250, geom_type="linestring")
        print(f" - added rura lint data in {time.time()-start:.2f}s")
        
        start=time.time()
        feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/bedrijfmilitaircamping2019_v2.shp"
        infer_gpd = add_rura_feature(infer_gpd, feature_path, feature_name="bmc", buffer_radius=250, geom_type="polygon")
        print(f" - added rura bmc data in {time.time()-start:.2f}s")
        
        ### STREETVIEW DATA

        # TODO: google streetview download + image analysis
        # Note: will require it be run in a different script with different environment
        #start=time.time()
        #file_path = os.path.dirname(__file__)
        #_command = f"conda run -n straatvinken-dl python {file_path}/01b-create-abt-streetview-analysis.py"
        #print(_command)
        # NOTE: should leave a file with given filename to join in next step!
        #print(f" - segmented streetview data in {time.time()-start:.2f}s")
        
        # join streetview results data
        start=time.time()
        infer_gpd = add_streetview_segmentdata(infer_gpd, segment_data_path="data/processed/20220325_streetview_coordinates_w_labels_quentin.pkl")
        #os.system(_command)
        print(f" - joined streetview landscape segmentation data in {time.time()-start:.2f}s")

        ### ITERATION 3
        start=time.time()
        infer_gpd = add_distance_data(infer_gpd, 
            featuredb_path=f"{BASE_DIR}belgium-latest-free.shp/gis_osm_traffic_free_1.shp",
            feature_col="trfeat", 
            feature_renames={"fclass": "trfeat"},
            neighbors=3)
        print(f" - added trfeat data in {time.time()-start:.2f}s")

        start=time.time()
        infer_gpd = add_distance_data(infer_gpd, 
            featuredb_path=f"{BASE_DIR}belgium-latest-free.shp/gis_osm_traffic_a_free_1.shp",
            feature_col="trfeatp", 
            feature_renames={"fclass": "trfeatp"},
            #feature_keepers=(),
            neighbors=3, 
            make_centroid=True)
        print(f" - added trfeatp distance data in {time.time()-start:.2f}s")

        start=time.time()
        infer_gpd = add_geometric_feature_availability(infer_gpd, 
            featuredb_path=f"{BASE_DIR}belgium-latest-free.shp/gis_osm_landuse_a_free_1.shp",
            feature_name="lu", 
            feature_renames={"fclass": "lu"},
            feature_keepers=("meadow", "forest", "farmland", "residential", "grass", "industrial", "park", "commercial"),
            feature_agglom=False, 
            verbose=False)
        # takes roughly 4000s :o
        print(f" - added lu data in {time.time()-start:.2f}s")
        ### END ITERATION 3

        start=time.time()
        infer_gpd = handle_missing_data(infer_gpd, most_frequent_imputer_columns=['verh', 'morf', 'wegcat'], report=True)
        print(f" - handled missing data in {time.time()-start:.2f}s")

        #TODO: data quality checks:
        # - is data point within flanders
        if infer_filetype == "pkl":
            infer_gpd.to_pickle(INFER_NO_OUTPUT_PATH)
        elif infer_filetype == "csv":
            infer_gpd.to_csv(INFER_NO_OUTPUT_PATH)
        print(f"data processing performed in {time.time()-start_start:.2f}s")

    start = time.time()
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
        with open(model_path, "rb") as model_file:
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

    print(f"complete script performed in {time.time()-start_start:.2f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="infer", description='Perform inference on a dataset.')
    parser.add_argument('--identifier', type=str, help='the identifier for the inference job', default="coordinates_quentin_large")
    parser.add_argument('--suffix', default="",help='file suffix')
    args = parser.parse_args()
    infer(args)