
import warnings


warnings.filterwarnings("ignore")

import os
import pandas as pd
import geopandas as gpd
import time
from datetime import date
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

pd.options.mode.chained_assignment = None

load_dotenv()

BASE_DIR="data/raw/"

def create_abt():
    first_ever = time.time()
    print("Creating ABT")

    start = time.time()
    sv_path=f"{BASE_DIR}straatvinken/SV2020_DataAll_20220211.csv"
    sv_gpd=get_straatvinken_data(sv_path)
    start_size=sv_gpd.shape[0]
    print(f" - loaded straatvinken dataset in {time.time()-start:.2f}s")

    start = time.time()
    bedi_path=f"{BASE_DIR}lu_bebdicht_ha_vlaa_2019_v2/lu_bebdicht_ha_vlaa_2019_v2.tif"
    sv_gpd = add_bebouwingsdichtheid(
        sv_gpd,
        bedi_path
    )
    print(f" - added bedi in {time.time()-start:.2f}s")

    start = time.time()
    statsect_path=f"{BASE_DIR}sh_statbel_statistical_sectors_20210101.shp/sh_statbel_statistical_sectors_20210101.shp"
    pode_path=f"{BASE_DIR}population_density_statbel/OPENDATA_SECTOREN_2021.xlsx"
    sv_gpd = add_populationdensity(
        sv_gpd, 
        pode_path, 
        statsect_path
    )
    print(f" - added pode in {time.time()-start:.2f}s")

    start = time.time()
    ncars_path = f"{BASE_DIR}number_of_cars_statbel/TF_CAR_HH_SECTOR.xlsx"
    sv_gpd = add_numberofcars(sv_gpd, ncars_path)
    print(f" - added ncars in {time.time()-start:.2f}s")

    start = time.time()
    popu_path=f"{BASE_DIR}TF_POPULATION_GRID_3035_20200101.shp/TF_POPULATION_GRID_3035_20200101.shp"
    sv_gpd = add_pop_sm(
        sv_gpd, 
        popu_path=popu_path
    )
    print(f" - added popu in {time.time()-start:.2f}s")

    start = time.time()
    acc_path = f"{BASE_DIR}traffic_accidents_statbel/TF_ACCIDENTS_2020.xlsx"
    sv_gpd = add_trafficaccidents(
        sv_gpd, 
        acc_path=acc_path,
        statsect_path=statsect_path
    )
    print(f" - added acc in {time.time()-start:.2f}s")

    start = time.time()
    wrsegm_path = f"{BASE_DIR}Wegenregister_SHAPE_20220317/Shapefile/Wegsegment.shp"
    wrsegm_gpd = get_wegsegment_data(wrsegm_path)
    print(f" - loaded wegsegment data in {time.time()-start:.2f}s")

    start = time.time()
    dbf_basedir = f"{BASE_DIR}Wegenregister_SHAPE_20220317/Shapefile/"
    wrsegm_gpd = add_wegsegment_attributes(wrsegm_gpd, dbf_basedir)
    print(f" - added wegsegment attributes in {time.time()-start:.2f}s")

    start = time.time()
    sv_gpd = add_wegsegment(sv_gpd, wrsegm_gpd)
    print(f" - added wr segment data in {time.time()-start:.2f}s")

    first_start = time.time()

    for max_distance in [50, 250, 1000]:
        start=time.time()
        sv_gpd = get_featurecount_within_distance(
            src_gdf=sv_gpd,
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
        sv_gpd = get_featurecount_within_distance(
            src_gdf=sv_gpd,
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
    school_path = f"{BASE_DIR}Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service_Shapefile/Shapefile/POI_Onderwijs.shp"
    sv_gpd = add_school_distance_data(sv_gpd, school_path, neighbors=3)
    print(f" - added school distance data in {time.time()-start:.2f}s")

    ### RURA DATA
    start=time.time()
    feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/kernclusters_kernen2019_v2.shp"
    sv_gpd = add_rura_feature(sv_gpd, feature_path, feature_name="kc_kern", buffer_radius=250, geom_type="polygon")
    print(f" - added rura kc_kern data in {time.time()-start:.2f}s")

    start=time.time()
    feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/kernclusters_linten2019_v2.shp"
    sv_gpd = add_rura_feature(sv_gpd, feature_path, feature_name="kc_lint", buffer_radius=250, geom_type="linestring")
    print(f" - added rura kc_lint data in {time.time()-start:.2f}s")
    
    start=time.time()
    feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/kernen2019_v2.shp"
    sv_gpd = add_rura_feature(sv_gpd, feature_path, feature_name="kern", buffer_radius=250, geom_type="polygon")
    print(f" - added rura kern data in {time.time()-start:.2f}s")
    
    start=time.time()
    feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/linten2019_v2.shp"
    sv_gpd = add_rura_feature(sv_gpd, feature_path, feature_name="lint", buffer_radius=250, geom_type="linestring")
    print(f" - added rura lint data in {time.time()-start:.2f}s")
    
    start=time.time()
    feature_path = f"{BASE_DIR}lu_klv_vlaa_2019/bedrijfmilitaircamping2019_v2.shp"
    sv_gpd = add_rura_feature(sv_gpd, feature_path, feature_name="bmc", buffer_radius=250, geom_type="polygon")
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
    sv_gpd = add_streetview_segmentdata(sv_gpd, segment_data_path="data/processed/20220302_streetview_coordinates_w_labels.pkl")
    #os.system(_command)
    print(f" - joined streetview landscape segmentation data in {time.time()-start:.2f}s")

    ### ITERATION 3
    start=time.time()
    sv_gpd = add_distance_data(sv_gpd, 
        featuredb_path=f"{BASE_DIR}osm-belgium-latest-free.shp/gis_osm_traffic_free_1.shp",
        feature_col="trfeat", 
        feature_renames={"fclass": "trfeat"},
        neighbors=3)
    print(f" - added trfeat data in {time.time()-start:.2f}s")

    start=time.time()
    sv_gpd = add_distance_data(sv_gpd, 
        featuredb_path=f"{BASE_DIR}osm-belgium-latest-free.shp/gis_osm_traffic_a_free_1.shp",
        feature_col="trfeatp", 
        feature_renames={"fclass": "trfeatp"},
        #feature_keepers=(),
        neighbors=3, 
        make_centroid=True)
    print(f" - added trfeatp distance data in {time.time()-start:.2f}s")

    start=time.time()
    sv_gpd = add_geometric_feature_availability(sv_gpd, 
        featuredb_path=f"{BASE_DIR}osm-belgium-latest-free.shp/gis_osm_landuse_a_free_1.shp",
        feature_name="lu", 
        feature_renames={"fclass": "lu"},
        feature_keepers=("meadow", "forest", "farmland", "residential", "grass", "industrial", "park", "commercial"),
        feature_agglom=False, 
        verbose=False)
    # takes roughly 4000s :o
    print(f" - added lu data in {time.time()-start:.2f}s")
    ### END ITERATION 3

    start=time.time()
    sv_gpd = handle_missing_data(sv_gpd, most_frequent_imputer_columns=['verh', 'morf', 'wegcat'], report=True)
    print(f" - handled missing data in {time.time()-start:.2f}s")

    start=time.time()
    final_size=sv_gpd.shape[0]
    print(f"final dataset size {final_size}")
    if start_size != final_size:
        export_filepath=f"data/processed/{date.today().isoformat().replace('-', '')}_straatvinken_abt_complete_df_CORRUPT.pkl"
        sv_gpd.to_pickle(export_filepath)
        raise ValueError(f"final size {final_size} is different from original size {start_size}!")
    else:
        export_filepath=f"data/processed/{date.today().isoformat().replace('-', '')}_straatvinken_abt_complete_df.pkl"
        sv_gpd.to_pickle(export_filepath)
    
    print(f"wrote final dataset in {time.time()-start:.2f}s to {export_filepath}")
    print(f"complete creation of abt took {time.time()-first_ever:.2f}s")

if __name__ == "__main__":
    create_abt()