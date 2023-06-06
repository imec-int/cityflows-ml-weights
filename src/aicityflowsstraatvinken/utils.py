import pandas as pd
import geopandas as gpd
import georasters as gr
from pyproj.crs import CRS
import osgeo
from osgeo.osr import SpatialReference
from scipy.spatial import cKDTree
import time
from typing import Tuple
import numpy as np
import itertools
from operator import itemgetter
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from simpledbf import Dbf5
import os
import warnings
import pickle
from glob import glob
from sklearn.impute import SimpleImputer
import yaml

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline #, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn import set_config
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression

BASE_DIR="data/raw/"

DATASET_BASE_EXPR="data/processed/*_straatvinken_abt_complete_df.pkl"

def get_straatvinken_data(sv_path = f"{BASE_DIR}straatvinken/SV2020_DataAll_20220211.csv", verbose=True):
    sv = pd.read_csv(sv_path, encoding = "ISO-8859-1")
    sv.columns = ["ID", "truck", "bus", "van", "car", "bike", "walk", "streetname", "municipality", "lat", "long"]
    sv_gpd = gpd.GeoDataFrame(sv, geometry=gpd.points_from_xy(sv.long, sv.lat), crs=4326)
    if verbose:
        print("straatvinken dataset initial size:", sv.shape[0])
        print("columns:", list(sv.columns))
    return sv_gpd

def add_bebouwingsdichtheid(gdf, tif_file=f"{BASE_DIR}lu_bebdicht_ha_vlaa_2013_v2/lu_bebdicht_ha_vlaa_2013_v2.tif"):
    raster_data=gr.from_file(tif_file)
    original_crs = gdf.crs
    if osgeo.version_info.major < 3:
        raster_crs = CRS.from_wkt(raster_data.projection.ExportToWkt())
    else:
        raster_crs = CRS.from_wkt(raster_data.projection.ExportToWkt(["FORMAT=WKT2_2018"]))
    gdf = gdf.to_crs(raster_crs)
    gdf["bedi"] = raster_data.map_pixel(gdf.geometry.x, gdf.geometry.y)
    #restore original geometry
    gdf = gdf.to_crs(original_crs)
    return gdf

def drop_col_if_exists(df, col = ("index_left", "index_right")):
    if type(col) == str:
        cols=[col]
    else:
        cols=col
    for col in cols:
        if col in df.columns:
            df = df.drop(columns=col)
    return df

def add_populationdensity(
    gdf, 
    pode_path = f"{BASE_DIR}population_density_statbel/OPENDATA_SECTOREN_2021.xlsx", 
    statsect_path=f"{BASE_DIR}Statistische_Sectoren_van_Belgie_20191218_Shapefile/Shapefile/Statsec.shp"
):
    pode_df = pd.read_excel(pode_path)
    statsect_gpd = gpd.read_file(statsect_path)
    statsect_gpd = statsect_gpd.to_crs(gdf.crs)
    pode_merged = pd.merge(statsect_gpd, pode_df, how="left", left_on="CS01012021", right_on="CD_SECTOR")
    pode_merged = pode_merged[["Shape_Leng", "Shape_Area", "TOTAL", "geometry"]]
    pode_merged.columns = ['ss_lengte', "ss_oppervl", "pode", "geometry"]
    gdf = drop_col_if_exists(gdf, col=("index_left", "index_right"))
    gdf = gdf.sjoin(pode_merged, how="left")
    gdf = drop_col_if_exists(gdf, col="index_right")
    return gdf

def add_numberofcars(
    gdf, 
    ncars_path = f"{BASE_DIR}number_of_cars_statbel/TF_CAR_HH_SECTOR.xlsx", 
    statsect_path = f"{BASE_DIR}sh_statbel_statistical_sectors_20210101.shp/sh_statbel_statistical_sectors_20210101.shp"
):
    """
    add_numberofcars:
        add the number of cars (ncars), number of households (nhh) and cars per household (ncars_hh)
        to a geopandas dataset, based on the statistical sector data from Flanders
    """
    ncars_df = pd.read_excel(ncars_path)
    statsect_gpd = gpd.read_file(statsect_path)
    statsect_gpd = statsect_gpd.to_crs(gdf.crs)
    ncars_merged = pd.merge(statsect_gpd, ncars_df, how="left", left_on="CS01012021", right_on="CD_STAT_SECTOR")
    ncars_merged = ncars_merged[["MS_NUM_HH", "MS_NUM_CAR", "geometry"]]
    ncars_merged.columns = ['nhh', "ncars", "geometry"]
    ncars_merged["ncars_hh"] = ncars_merged["ncars"] / ncars_merged["nhh"]
    gdf = drop_col_if_exists(gdf, col=("index_left", "index_right"))
    gdf = gdf.sjoin(ncars_merged, how="left")
    gdf = drop_col_if_exists(gdf, col="index_right")
    gdf.nhh = gdf.nhh.fillna(gdf.nhh.median())
    gdf.ncars = gdf.ncars.fillna(gdf.ncars.median())
    gdf.ncars_hh = gdf.ncars / gdf.nhh
    return gdf

def get_wegsegment_data(wrsegm_path):
    wrsegm_gdf = gpd.read_file(wrsegm_path)
    wrsegm_gdf = wrsegm_gdf[["WS_OIDN", "MORF", "WEGCAT", "geometry"]]
    wrsegm_gdf.columns = ["WS_OIDN", "morf", "wegcat", "geometry"]
    wrsegm_gdf[["morf", "wegcat"]] = wrsegm_gdf[["morf", "wegcat"]].astype(str)
    wrsegm_gdf["morf"] = wrsegm_gdf["morf"].apply(lambda x: x.replace(".0", ""))
    wrsegm_gdf["wegcat"] = wrsegm_gdf["wegcat"].apply(lambda x: x.replace(".0", ""))
    return wrsegm_gdf

def add_wegsegment_attributes(
    wrsegm_gdf,
    dbf_basepath = f"{BASE_DIR}Wegenregister_SHAPE_20211216/Shapefile/"):
    # route is E-weg
    atteuropweg_df = Dbf5(os.path.join(dbf_basepath, "AttEuropweg.dbf")).to_dataframe()
    atteuropweg_df["is_eurw"] = 1
    atteuropweg_df = atteuropweg_df[["WS_OIDN", "is_eurw"]]
    #atteuropweg_df
    # route is N-weg
    attnationweg_df = Dbf5(os.path.join(dbf_basepath, "AttNationweg.dbf")).to_dataframe()
    attnationweg_df["is_natw"] = 1
    attnationweg_df = attnationweg_df[["WS_OIDN", "is_natw"]]
    #attnationweg_df
    # aantal rijstroken
    attrijstroken_df = Dbf5(os.path.join(dbf_basepath, "AttRijstroken.dbf")).to_dataframe()
    attrijstroken_df = attrijstroken_df[(attrijstroken_df["AANTAL"] > 0)].groupby("WS_OIDN").agg({"AANTAL": sum}).reset_index()
    attrijstroken_df.columns=["WS_OIDN", "nrijstr"]
    #attrijstroken_df
    # weg verharding
    attwegverhard_df = Dbf5(os.path.join(dbf_basepath, "AttWegverharding.dbf")).to_dataframe()
    attwegverhard_df = attwegverhard_df[(attwegverhard_df.TYPE > 0)]
    attwegverhard_df["LEN"] = np.abs(attwegverhard_df["TOTPOS"] - attwegverhard_df["VANPOS"])
    attwegverhard_df = attwegverhard_df.sort_values(by=["WS_OIDN", "LEN"], ascending=[True, False])
    attwegverhard_df = attwegverhard_df.groupby("WS_OIDN").nth(0).reset_index()[["WS_OIDN", "TYPE", "LBLTYPE"]]
    attwegverhard_df["TYPE"] = attwegverhard_df["TYPE"].astype(str)
    attwegverhard_df["LBLTYPE"] = attwegverhard_df["LBLTYPE"].astype(str)
    attwegverhard_df.columns = ["WS_OIDN", "verh", "verhlbl"]
    attwegverhard_df
    # wegbreedte
    attwegbreedte_df = Dbf5(os.path.join(dbf_basepath, "AttWegbreedte.dbf")).to_dataframe()
    attwegbreedte_df = attwegbreedte_df[
        (attwegbreedte_df["BREEDTE"] > 0)
    ].groupby("WS_OIDN").agg({"BREEDTE": np.mean}).reset_index()
    attwegbreedte_df.columns = ["WS_OIDN", "wb"]
    attwegbreedte_df
    # genummerde weg ... not usefull?
    #attgenumweg_df = Dbf5(os.path.join(dbf_basepath, "AttGenumweg.dbf")).to_dataframe()
    # ongelijkgrondse kruising
    #attrltogkruising_df = Dbf5(os.path.join(dbf_basepath, "RltOgkruising.dbf")).to_dataframe()
    # Join to wrsegm_df
    start = time.time()
    wrsegm_gdf_att=wrsegm_gdf.copy()
    for attdf in [atteuropweg_df, attnationweg_df, attrijstroken_df, attwegverhard_df, attwegbreedte_df]:
        wrsegm_gdf_att = pd.merge(wrsegm_gdf_att, attdf, how="left", on="WS_OIDN")
    wrsegm_gdf_att[["is_eurw", "is_natw", "wb", "nrijstr"]] = wrsegm_gdf_att[["is_eurw", "is_natw", "wb", "nrijstr"]].fillna(0)
    print(f" - wr attributes joined to segments in {time.time() - start:.2f}s")
    return wrsegm_gdf_att

def add_wegsegment(
    gdf,
    wrsegm_gdf
):
    # convert to world mercator projection, distance unit is meters
    gpd_wr = gdf.to_crs(3395).sjoin_nearest(wrsegm_gdf.to_crs(3395), how="left", max_distance=10, distance_col="wegsegment_distance")
    
    gpd_wr["nrijstr"] = gpd_wr["nrijstr"].fillna(gpd_wr["nrijstr"].median())
    gpd_wr["wb"] = gpd_wr["wb"].fillna(gpd_wr["wb"].median())
    return gpd_wr.to_crs(4326)

def get_featurecount_within_distance(
    src_gdf: gpd.GeoDataFrame, 
    featuredb_gdf: gpd.GeoDataFrame, 
    feature_col: str,
    max_distance: float = 1000,
    max_neighbours: int = 350, 
    result_type: str = "feature_columns",
    column_suffix: str = "",
    total_buckets_prefix: Tuple[str, ...] = (),
    feature_filters: Tuple[str, ...] = ("-8"," -9"),
    verbose=False
):
    """
    Returns the number of a given 'feature column' occurence within a given distance

            Parameters:
                    src_gdf (GeoDataFrame): dataframe containing the coordinates 
                        to be enriched with the featurecounts
                    featuredb_gdf (GeoDataFrame): feature dataframe with linestrings for geometry objects
                    feature_col (str): the column of which occurence counts will be made
                    result_type (str): 
                        feature_columns: explode the categories into separate columns 
                            using the "feature_col"_"category" naming convention
                        map: add result as a dictionary object to a single column with the name "feature_col"

            Returns:
                    enriched_gdf (GeoDataFrame): the original dataframe with feature count data added
    
    """
    #TODO fix documentation
    
    if result_type not in ["feature_columns", "map"]:
        raise ValueError(f"Invalid result_type parameter value '{result_type}'")
        
    start = time.time()
    src_gdf=src_gdf.to_crs(3395)
    featuredb_gdf=featuredb_gdf.to_crs(3395)
    if verbose:
        print(f"crs adaptation performed in {time.time() - start:.2f}s")
    
    if len(feature_filters) > 0:
        start=time.time()
        featuredb_gdf = featuredb_gdf[(~featuredb_gdf[feature_col].isin(feature_filters))]
        if verbose:
            print(f"feature filter performed in {time.time() - start:.2f}s")
    
    zero_features = {key: 0 for key in set(featuredb_gdf[feature_col])}
    
    start = time.time()
    A = np.concatenate(
        [np.array(geom.coords) for geom in src_gdf.geometry.to_list()])
    B = [np.array(geom.coords) for geom in featuredb_gdf.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    if verbose:
        print(f"geometries created in {time.time() - start:.2f}s")
    
    max_neighbours = min(max_neighbours, B.shape[0])
    
    start = time.time()
    ckd_tree = cKDTree(B)
    if verbose:
        print(f"ckd tree created in {time.time() - start:.2f}s")
    
    start = time.time()
    dist, idx = ckd_tree.query(A, k=max_neighbours)
    if verbose:
        print(f"query performed in {time.time() - start:.2f}s")
    
    start = time.time()
    feature_counts = []
    num_neighbours = []
    for dist, idx in zip(dist, idx):
        idx = itemgetter(*idx)(B_ix)
        closest_gdf = pd.DataFrame(
            {
                "idx": list(idx), 
                feature_col: featuredb_gdf.iloc[list(idx)][feature_col].values, 
                "dist": dist
            }
        )
        closest_gdf = closest_gdf[eval(f"dist < {max_distance}")]
        cleanidx = list(dict.fromkeys(closest_gdf["idx"].values))
        feature_count = zero_features.copy()
        feature_count.update(dict(featuredb_gdf.iloc[cleanidx][feature_col].value_counts()))
        
        #add totals
        for total_bucket in total_buckets_prefix:
            feature_count[f"{total_bucket}tot"] = sum([feature_count[key] for key in feature_count.keys() if key.startswith(total_bucket)])
        
        feature_counts.append(feature_count)
        num_neighbours.append(len(cleanidx))
    
    if verbose:
        print(f"data adaptation performed in {time.time() - start:.2f}s")
    if result_type == "feature_columns":
        features_df = pd.DataFrame(feature_counts)
        features_df.columns = [f"{feature_col}_{col}{column_suffix}" for col in features_df.columns]
        gdf = pd.concat(
            [src_gdf, features_df, pd.Series(num_neighbours, name=f"{feature_col}{column_suffix}_num_neighbors")], axis=1)
        return gdf.to_crs(4326)
    else:
        gdf = pd.concat(
            [src_gdf, 
             pd.Series(feature_counts, name=f"{feature_col}{column_suffix}_vicinity_counts"),
             pd.Series(num_neighbours, name=f"{feature_col}{column_suffix}_num_neighbors")], axis=1)
        return gdf.to_crs(4326)

def add_pop_sm(
    gdf, 
    popu_path = f"{BASE_DIR}TF_POPULATION_GRID_3035_20200101.shp/TF_POPULATION_GRID_3035_20200101.shp", 
):
    popu_gpd = gpd.read_file(popu_path)
    popu_gpd = popu_gpd.to_crs(gdf.crs)
    popu_gpd = popu_gpd[["ms_pop", "geometry"]]
    gdf = gdf.sjoin(popu_gpd, how="left")
    gdf = drop_col_if_exists(gdf, col="index_right")
    return gdf

def add_attributes(
    df: pd.DataFrame,
    attdf: pd.DataFrame,
    on,
    how: str,
    filtr: str=None,
    retain_cols: Tuple[str, ...]=None
):
    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError("'how' parameter doesn't have one of the accepted values: left, right, outer, inner")
    
    if filtr is not None:
        attdf = attdf[attdf.eval(filtr)]
    if retain_cols is not None:
        retc = list(retain_cols)
        try:
            attdf[retc]
        except KeyError as ke:
            raise ValueError(f"one of required columns does not exist: {ke}")
    if type(on) == str:
        if not(on in df.columns and on in attdf.columns):
            raise ValueError("'on' column doesn't exist in one of the provided dataframes")    
        if retain_cols is not None:
            retc = list(retain_cols)
            retc.append(on)
            attdf = attdf[retc]
        # check attdf by grouping on "on"
        if True in list((attdf.groupby(on).size()>1).value_counts().keys()):
            warnings.warn("attribute dataframe has multiple rows when grouping on the 'on' column")
        df = pd.merge(df, attdf, how=how, on=on)
    elif type(on) in [tuple, list]:
        if len(on) != 2:
            raise ValueError("'on' tuple/list doesn't have exactly 2 items")
        left_on, right_on = on
        # check attdf by grouping on "right_on"
        if True in list((attdf.groupby(right_on).size()>1).value_counts().keys()):
            warnings.warn("attribute dataframe has multiple rows when grouping on the rightmost 'on' column")
        if retain_cols is not None:
            retc = list(retain_cols)
            retc.append(right_on)
            attdf = attdf[retc]
        df = pd.merge(df, attdf, how=how, left_on=left_on, right_on=right_on)
    else:
        raise ValueError(f"'on' has unacceptable type {type(on)} (should be in tuple, list or str)")
    return df

def add_trafficaccidents(
    gdf, 
    acc_path = f"{BASE_DIR}traffic_accidents_statbel/TF_ACCIDENTS_2020.xlsx",
    statsect_path=f"{BASE_DIR}sh_statbel_statistical_sectors_20210101.shp/sh_statbel_statistical_sectors_20210101.shp"
):
    """
    add_trafficaccidents:
        add the number of traffic accidents to a geopandas dataset
        based on the statistical sector data from Flanders
    """
    acc_df = pd.read_excel(acc_path)
    acc_df = acc_df.rename(columns={
        "MS_ACCT": "acc",
        "MS_ACCT_WITH_DEAD": "acc_death",
        "MS_ACCT_WITH_DEAD_30_DAYS": "acc_death30",
        "MS_ACCT_WITH_MORY_INJ": "acc_mort",
        "MS_ACCT_WITH_SERLY_INJ": "acc_ser",
        "MS_ACCT_WITH_SLY_INJ": "acc_sly"
    })
    counter_cols = ["acc", "acc_death", "acc_death30", "acc_mort", "acc_ser", "acc_sly"]
    acc_df["CD_MUNTY_REFNIS"] = acc_df["CD_MUNTY_REFNIS"].astype(str)
    counter_cols.append("CD_MUNTY_REFNIS")
    acc_df = acc_df[counter_cols].groupby("CD_MUNTY_REFNIS").agg("sum").reset_index()
    statsect_gpd = gpd.read_file(statsect_path)
    statsect_gpd = statsect_gpd.to_crs(gdf.crs)
    statsect_gpd = add_attributes(
        statsect_gpd, 
        acc_df, 
        how="left", 
        on=["CNIS5_2021", "CD_MUNTY_REFNIS"], 
        retain_cols=["acc", "acc_death", "acc_death30", "acc_mort", "acc_ser", "acc_sly"]
    )
    statsect_gpd = statsect_gpd[["geometry", "acc", "acc_death", "acc_death30", "acc_mort", "acc_ser", "acc_sly"]]
    gdf = drop_col_if_exists(gdf, col=["index_left", "index_right"])
    gdf = gdf.sjoin(statsect_gpd, how="left")
    gdf = drop_col_if_exists(gdf, col=["index_left", "index_right"])
    return gdf


def add_streetview_segmentdata(sv_gpd, segment_data_path="data/processed/20220302_streetview_coordinates_w_labels.pkl"):
    streetview_segmentation_df = pickle.load(open(segment_data_path, "rb"))
    # apparently this was a problem :/
    streetview_segmentation_df = streetview_segmentation_df.drop_duplicates()
    landscape_rename = {
        col: f"segm_{col}" for col in streetview_segmentation_df.select_dtypes(float).columns 
        if col not in ["lat", "long"]
    }
    #make decimal number (between 0 and 1) 
    # instead of between 0 and 100
    streetview_segmentation_df[list(landscape_rename.keys())] = streetview_segmentation_df[list(landscape_rename.keys())] / 100
    streetview_segmentation_df = streetview_segmentation_df.rename(columns=landscape_rename)
    segm_gpd = gpd.GeoDataFrame(streetview_segmentation_df, geometry=gpd.points_from_xy(streetview_segmentation_df.long, streetview_segmentation_df.lat), crs=4326)
    sv_gpd = drop_col_if_exists(sv_gpd, col=("index_left", "index_right"))
    segm_gpd = drop_col_if_exists(segm_gpd, col=("index_left", "index_right"))
    sv_segm_gpd = sv_gpd.to_crs(3395).sjoin_nearest(segm_gpd.to_crs(3395), how="left", max_distance=50, distance_col="segm_distance")
    landscape_segm_cols = list(landscape_rename.values())
    sv_segm_gpd = drop_col_if_exists(sv_segm_gpd, col=("index_left", "index_right", 'lat_right', 'long_right'))
    sv_segm_gpd = sv_segm_gpd.rename(columns={'lat_left': "lat", 'long_left': "long"})
    #report_missing_data(sv_segm_gpd[landscape_segm_cols])
    sv_segm_gpd[landscape_segm_cols] = sv_segm_gpd[landscape_segm_cols].fillna(0)
    return sv_segm_gpd.to_crs(4326)


def get_avg_distance_of_k_neighbors(
    src_gdf: gpd.GeoDataFrame, 
    featuredb_gdf: gpd.GeoDataFrame, 
    neighbors: int = 25, 
    feature_col: str="school",
    feature_keepers: Tuple[str, ...] = (),
    feature_filters: Tuple[str, ...] = (),
    verbose=False
):
    """
    Returns the average distance of the k neareest neightbors subdivided to a given 'feature column'

            Parameters:
                    src_gdf (GeoDataFrame): dataframe containing the coordinates 
                        to be enriched with the featurecounts
                    featuredb_gdf (GeoDataFrame): feature dataframe with POI's for geometry objects
                    feature_col (str): the column of which occurence counts will be made
                    result_type (str): 
                        feature_columns: explode the categories into separate columns 
                            using the "feature_col"_"category" naming convention
                        map: add result as a dictionary object to a single column with the name "feature_col"

            Returns:
                    enriched_gdf (GeoDataFrame): the original dataframe with feature count data added
    
    """
    
    start = time.time()
    # this CRS allows for measuring distance in meters
    src_gdf=src_gdf.to_crs(3395)
    featuredb_gdf=featuredb_gdf.to_crs(3395)
    if verbose:
        print(f"crs adaptation performed in {time.time() - start:.2f}s")
    
    if len(feature_keepers) > 0:
        if len(feature_filters) > 0:
            print("both feature_keepers and feature_filters were set, only applying feature keepers")
        start=time.time()
        featuredb_gdf = featuredb_gdf[(featuredb_gdf[feature_col].isin(feature_keepers))]
        if verbose:
            print(f"feature keeper performed in {time.time() - start:.2f}s")
    elif len(feature_filters) > 0:
        start=time.time()
        featuredb_gdf = featuredb_gdf[(~featuredb_gdf[feature_col].isin(feature_filters))]
        if verbose:
            print(f"feature filter performed in {time.time() - start:.2f}s")
    
    start = time.time()
    
    nA = np.array(list(src_gdf.geometry.apply(lambda x: (x.x, x.y))))
    
    
    if verbose:
        print(f"source geometries created in {time.time() - start:.2f}s")
    
    feature_counts = {}
    
    for feature in pd.unique(featuredb_gdf[feature_col]):
        start = time.time()
        frag = featuredb_gdf[featuredb_gdf.eval(f"{feature_col} == '{feature}'")]
        
        nB = np.array(list(frag.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        
        if verbose:
            print(f"ckd tree created for feature {feature} in {time.time() - start:.2f}s")
    
        start = time.time()
        max_neighbors = min(neighbors, nB.shape[0])
        dist, idx = btree.query(nA, k=max_neighbors)
        if verbose:
            print(f"query for feature {feature} performed in {time.time() - start:.2f}s")
    
        start = time.time()
        #gdB_nearest = frag.iloc[idx].drop(columns="geometry").reset_index(drop=True)
        feature_avg_dist=[]
        #nn = []
        for dist, idx in zip(dist, idx):
            #nn.append(len(list(idx)))
            feature_avg_dist.append(np.median(dist))
        #feature_counts[f"{feature}_num_neighbors"] = nn
        feature_counts[f"{feature}_dist"] = feature_avg_dist
    
    if verbose:
        print(f"data adaptation performed in {time.time() - start:.2f}s")
    
    features_df = pd.DataFrame(feature_counts)
    features_df.columns = [f"{feature_col}_{col}{neighbors}" for col in features_df.columns]
    gdf = pd.concat([src_gdf, features_df], axis=1)
    return gdf.to_crs(4326)


def add_school_distance_data(sv_gpd:gpd.GeoDataFrame, 
    school_path:str=f"{BASE_DIR}Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service_Shapefile/Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service/Shapefile/POI_Onderwijs.shp", 
    neighbors:int=5):
    school_df = gpd.read_file(school_path)
    school_df["school"]=school_df["CATEGORIE"].apply(lambda x: x[0])
    return get_avg_distance_of_k_neighbors(sv_gpd, school_df, feature_col="school", neighbors=neighbors)

def add_distance_data(sv_gpd:gpd.GeoDataFrame, 
    featuredb_path:str=f"{BASE_DIR}Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service_Shapefile/Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service/Shapefile/POI_Onderwijs.shp", 
    feature_col="fclass",
    neighbors:int=3,
    feature_renames:dict={},
    feature_keepers: Tuple[str, ...]= (),
    feature_filters: Tuple[str, ...]= (),
    make_centroid:bool=False
    ):
    feature_df = gpd.read_file(featuredb_path)
    if len(list(feature_renames.keys())) > 0:
        feature_df = feature_df.rename(columns=feature_renames)
    
    if len(feature_keepers) > 0:
        if len(feature_filters) > 0:
            print("both feature_keepers and feature_filters were set, only applying feature keepers")
        featuredb_gpd = featuredb_gpd[(featuredb_gpd[feature_col].isin(feature_keepers))]
    elif len(feature_filters) > 0:
        featuredb_gpd = featuredb_gpd[(~featuredb_gpd[feature_col].isin(feature_filters))]
    
    if make_centroid:
        #feature_df["poly"] = feature_df.geometry
        feature_df.geometry = feature_df.geometry.centroid
    return get_avg_distance_of_k_neighbors(sv_gpd, feature_df, feature_col=feature_col, neighbors=neighbors)
    
#TODO: rename to more generic function name (also used for LU)
def add_rura_feature(
    sv_gpd: gpd.GeoDataFrame,
    featuredb_path: str,
    feature_name: str,
    feature_renames: dict={},
    buffer_radius: int=250,
    geom_type: str="polygon"
    ):
    if geom_type not in ["polygon", "linestring"]:
        raise ValueError(f"geom_type must be one of 'polygon', 'linestring' but is '{geom_type}'")
    featuredb_gpd = gpd.read_file(featuredb_path)
    if len(list(feature_renames.keys()))> 0:
        featuredb_gpd = featuredb_gpd.rename(columns=feature_renames)
    sv_gpd = sv_gpd.to_crs(3395)
    featuredb_gpd = featuredb_gpd.to_crs(3395)
    feature = featuredb_gpd.unary_union
    circles = sv_gpd.buffer(buffer_radius)
    sv_gpd[feature_name] = circles.intersection(feature)
    if geom_type=="linestring":
        sv_gpd[f"{feature_name}_len"]=sv_gpd[feature_name].length
    else: #  type=="polygon"
        sv_gpd[f"{feature_name}_prop"]=sv_gpd[feature_name].area / circles.area
    return sv_gpd.to_crs(4326)

def add_geometric_feature_availability(
    sv_gpd: gpd.GeoDataFrame,
    featuredb_path: str,
    feature_name: str,
    feature_renames: dict={},
    feature_keepers: Tuple[str, ...]= (),
    feature_filters: Tuple[str, ...]= (),
    feature_agglom: bool=True,
    buffer_radius: int=250,
    geom_type: str="polygon",
    verbose:bool=False
    ):
    if geom_type not in ["polygon", "linestring"]:
        raise ValueError(f"geom_type must be one of 'polygon', 'linestring' but is '{geom_type}'")
    featuredb_gpd = gpd.read_file(featuredb_path)

    if len(list(feature_renames.keys()))> 0:
        featuredb_gpd = featuredb_gpd.rename(columns=feature_renames)
    sv_gpd = sv_gpd.to_crs(3395)
    featuredb_gpd = featuredb_gpd.to_crs(3395)
    if len(feature_keepers) > 0:
        if len(feature_filters) > 0:
            print("both feature_keepers and feature_filters were set, only applying feature keepers")
        featuredb_gpd = featuredb_gpd[(featuredb_gpd[feature_name].isin(feature_keepers))]
    elif len(feature_filters) > 0:
        featuredb_gpd = featuredb_gpd[(~featuredb_gpd[feature_name].isin(feature_filters))]
    
    if feature_agglom:
        feature = featuredb_gpd.unary_union
        circles = sv_gpd.buffer(buffer_radius)
        sv_gpd[feature_name] = circles.intersection(feature)
        if geom_type=="linestring":
            sv_gpd[f"{feature_name}_len"]=sv_gpd[feature_name].length
        else: #  type=="polygon"
            sv_gpd[f"{feature_name}_prop"]=sv_gpd[feature_name].area / circles.area
    else:
        features= pd.unique(featuredb_gpd[feature_name])
        for ix, feature in enumerate(features):
            feature_geom = featuredb_gpd[featuredb_gpd[feature_name]==feature].unary_union
            circles = sv_gpd.buffer(buffer_radius)
            sv_gpd[f"{feature_name}_{feature}"] = circles.intersection(feature_geom)
            if geom_type=="linestring":
                sv_gpd[f"{feature_name}_{feature}_len"]=sv_gpd[f"{feature_name}_{feature}"].length
            else: #  type=="polygon"
                sv_gpd[f"{feature_name}_{feature}_prop"]=sv_gpd[f"{feature_name}_{feature}"].area / circles.area
            if verbose:
                print(f"    added feature stat for '{feature}' ({ix+1}/{len(features)})")
    return sv_gpd.to_crs(4326)

def report_missing_data(dataset):
    null_mask = dataset.isnull().sum()
    df_report = null_mask[null_mask.index[(null_mask>0)]]
    if df_report.shape[0] == 0:
        print(f"no missing data")
    else:
        print(f"missing data:{df_report.to_string()}")

def handle_missing_data(dataset, most_frequent_imputer_columns = None, report=False):
    transformer = SimpleImputer(strategy='most_frequent')
    if most_frequent_imputer_columns:
        dataset[most_frequent_imputer_columns] = transformer.fit_transform(dataset[most_frequent_imputer_columns])
    if report:
        report_missing_data(dataset)
    return dataset

def most_recent_dataset(dataset_path_expr=DATASET_BASE_EXPR):
    datasets = glob(dataset_path_expr)
    return sorted(datasets)[-1]

def get_traintest_data(dataset: gpd.GeoDataFrame, config: dict, random_state:int=42, stratify:str=None):
    sv_wide_df = dataset
    # load column config
    print(f"config iteration {config['iteration']} created on {config['date_created']}")
    Y_s = config["columns"]["Y_s"]
    explain_cols = config["columns"]["explain_cols"]
    one_hot_cols = config["columns"]["one_hot_cols"]
    num_pred_remain_cols = config["columns"]["num_pred_remain_cols"]
    num_pred_segm_cols = config["columns"]["num_pred_segm_cols"]
    num_pred_minmax_cols = config["columns"]["num_pred_minmax_cols"]

    # remove outliers in y's: only those below 99% percentile retained
    sv_wide_df = sv_wide_df.iloc[sv_wide_df.index[
        np.all(sv_wide_df[Y_s] < np.percentile(sv_wide_df[Y_s]
        , 99, axis=0), axis=1)]
    ]

    # deal with missing data
    sv_wide_df.nhh = sv_wide_df.nhh.fillna(sv_wide_df.nhh.median())
    sv_wide_df.ncars = sv_wide_df.ncars.fillna(sv_wide_df.ncars.median())
    sv_wide_df.ncars_hh = sv_wide_df.ncars / sv_wide_df.nhh

    # create train and test dataset
    X_cols = explain_cols.copy()
    X_cols.extend(one_hot_cols)
    X_cols.extend(num_pred_minmax_cols)
    X_cols.extend(num_pred_segm_cols)
    X_cols.extend(num_pred_remain_cols)

    if stratify: 
        return train_test_split(
            sv_wide_df[X_cols], 
            sv_wide_df[Y_s], 
            test_size=.2, 
            random_state=random_state,
            shuffle=True,
            stratify=sv_wide_df[stratify]
        )
    else:
        return train_test_split(
            sv_wide_df[X_cols], 
            sv_wide_df[Y_s], 
            test_size=.2, 
            random_state=random_state,
            shuffle=True
        )

class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.names]

def prepare_model(estimator, config, params, scoring, refit):
    one_hot_cols = config["columns"]["one_hot_cols"]
    num_pred_remain_cols = config["columns"]["num_pred_remain_cols"]
    num_pred_segm_cols = config["columns"]["num_pred_segm_cols"]
    num_pred_minmax_cols = config["columns"]["num_pred_minmax_cols"]
    one_hot_col_names = config["columns"]["one_hot_col_names"]
    one_hot_categories = config["columns"]["one_hot_categories"]

    num_pred_minmax = num_pred_minmax_cols.copy()
    num_pred_minmax.extend(num_pred_segm_cols)

    cols = one_hot_col_names.copy()
    cols.extend(num_pred_minmax)
    cols.extend(num_pred_remain_cols)
        
    num_pred_minmax = num_pred_minmax_cols.copy()
    num_pred_minmax.extend(num_pred_segm_cols)

    _pipeline_def = ("features", FeatureUnion([
        ('ohe', make_pipeline(
            Columns(names=one_hot_cols),
            OneHotEncoder(sparse=False, drop="first", categories=one_hot_categories, handle_unknown="error"))),
        ('mima', make_pipeline(
            Columns(names=num_pred_minmax),
            MinMaxScaler())),
        ('keep', make_pipeline(Columns(names=num_pred_remain_cols)))
    ]))

    pipe = Pipeline(
        [
            _pipeline_def,
            ('est', estimator)
        ]
    )
    return GridSearchCV(pipe, params, scoring=scoring, refit=refit, cv=10, verbose=0)

def accuracy_report(model, X, y, summary=False):
    if summary:
        return r2_score(y,model.predict(X))
    else:
        return f"""
        R2:              {r2_score(y,model.predict(X)):8.4f}
        Pearson corrcoef:{np.corrcoef(y, model.predict(X))[0, 1]:8.4f}
        RMSE:            {np.sqrt(mean_squared_error(y,model.predict(X))):8.4f}
        MAE:             {mean_absolute_error(y,model.predict(X)):8.4f}
        """

def report_model( gridsearch_result, X_train, y_train,  X_test, y_test, config):
    model = gridsearch_result.best_estimator_.named_steps["est"]
    model_name = type(model).__name__
    return f"""#######################################################
#### report of model '{model_name}' training ####
#######################################################

model data
------------------
{yaml.dump(config['model'], sort_keys=False, default_flow_style=False)}

train set accuracy
------------------
{accuracy_report(gridsearch_result, X_train, y_train)}

test set accuracy
------------------
{accuracy_report(gridsearch_result, X_test, y_test)}"""





