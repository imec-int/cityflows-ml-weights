# cityflows-ml-weights

Predicting traffic intensity from geospatial attributes

## Setup
1. Install git and checkout the [git code repository]
2. Install [anaconda] python version 3.8+
3. Change working directory into the git code repository root
4. Run all things automateable by using

   `make all`

Note: separate parts (or as a fallback after manual steps) can be run by inspecting the Makefile and using as needed


## Using the Python Conda environment

Create the conda environment using: `make env`

Once the Python Conda environment has been set up, you can

* Activate the environment using the following command in a terminal window:

  * Windows: `activate straatvinken`
  * Linux, OS X: `source activate straatvinken`
  * The __environment is activated per terminal session__, so you must activate it every time you open terminal.

* Deactivate the environment using the following command in a terminal window:

  * Windows: `deactivate straatvinken`
  * Linux, OS X: `source deactivate straatvinken`

Note: there is also a straatvinken-dl environment, which can be installed for running AI specific tasks (segmentation of Google Streetview images)
               
Delete the environment using the command (can't be undone): `make rmenv`

## Running independent data science steps

As can also be discovered by exploring the Makefile, one can run the following commands to perform separate steps in the data science pipeline:

* `make env`: create conda environment
* `make download`: create data filestructure & download all relevant source files (note: some parts are not fully automatic, namely sync-ing the data from Azure, which may have to be done manually)
* `make extract`: extract all the archives that were downloaded
* `make abt`: create the analytical base table (abt) based on all the source data
* `make train`: train the models (includes hyperparameter tuning) based on the abt
* `make infer`: perform inference on a given set of coordinates

## Data science workflow

As already became apparent in the Makefile structure, there's a certain flow in the data science project. 

__0. Environment creation__
First, a python environment is created, using Conda, the relevant packages are in `conda_env.yaml` - which needs to be maintained as the project evolves.

_Important note:_ If you wish to use Google Streetview data, you should create a `.env` file in the root of the solution with the following content:

```
export GOOGLE_API_KEY=<INSERT_KEY>
```

__1. Data download & extraction__
The datasets in this project come from a number of sources, most of which are freely accessible (open data). It has occured in the past that links are taken down, changed or updated data files are provided (with varying data schema's or filenames within the archives). Now that the project has ended these will no longer be updated, however the substitute dataset should easily be found from the websites they come from.

Some files require a manual process of registering on a web portal, putting the (free) dataset in a shopping cart, and waiting for the data providing authority's approval to provide you the link (using an authentication procedure). When this is the case, it is mentioned in the comments of the Makefile.

__2. Analytical base table creation__

In this project, many datasets that are geospatial in nature are joined to the original straatvinken coordinates. A number of utility functions were created to facilitate this process.

_Intermezzo: geospatial data files & CRS_
There are a number of different data types to be found:
* __Shapefiles__: The shapefile format is a geospatial vector data format for geographic information system (GIS) software. It was created and is regulated by Esri. Several files are present in a single directory - which for data sharing purposes is typically archived in a zipfile. Shapefiles are read by Geopandas by providing the path to the file with the `.shp` extension.
* __Raster files__: A raster file is a file which captures geospatial data in a grid-layout (with rows and columns). Each such a cell in this grid contains one or more numerical values. The raster files in this project are all of the GeoTiff standard and have the `.tiff` extension.
*  __CSV's & Excel files__ : 
_comma separated value_ files make use of a separator (typically a comma, but sometimes it's a tab or a semi-colon), and have the `.csv` extension.
_excel files_ follow Microsoft's proprietary format, most of them have the XLSX extension.
Both filetypes house a columnar data format, in which no geospatial notion exists. This means that the latitude and longitude are coded as floating point numbers, and the correct geometry has to be created from them (including the appropriate `CRS - coordinate reference system`, which is typically `EPSG:4326`)

Similarly, several `CRS - coordinate reference system`s exist. These represent the way that coordinates on the globe are represented on a flat surface (which has to deal with earth curvature etc). 

Common CRS's in this solution are:
* [EPSG:4326 WGS 84 -- WGS84 - World Geodetic System 1984, used in GPS](https://epsg.io/4326)
* [EPSG:31370 Belge 1972 / Belgian Lambert 72 -- Belgium](https://epsg.io/31370): high precision coordinate system for Belgium
* [EPSG:3395 WGS 84 / World Mercator](https://epsg.io/3395): handy because the unit of measurement is in metres (so can be used when distance between 2 points needs to be calculated)

## Initial File Structure

```
├── .gitignore               <- Files that should be ignored by git.
├── conda_env.yml            <- Conda environment definition
├── conda_env_dl.yml         <- Conda deep learning environment
├── conda_env_minimal.yml    <- Conda demo environment
├── LICENSE
├── README.md                <- This README
├── requirements.txt         <- The requirements file for reproducing the analysis environment
│                               generated with `pip freeze > requirements.txt`. Might not be needed if using conda.
├── setup.py                 <- Metadata about project for easy distribution.
│
├── data
│   ├── external             <- External immutable files, unrelated to analytics effort
│   ├── processed            <- The final, canonical data sets for modeling.
│   ├── raw                  <- The original, immutable data dump.
│   ├── temp                 <- Temporary files.
│   └── training             <- Files relating to the training process
│
├── docs                     <- Documentation (NOT USED)
│   ├── data_science_code_of_conduct.md  <- Code of conduct
│   ├── process_documentation.md         <- Standard template for documenting process and decisions.
│   └── writeup              <- Sphinx project for project writeup including auto generated API.
│      ├── conf.py           <- Sphinx configuration file
│      ├── index.rst         <- Start page.
│      ├── make.bat          <- For generating documentation (Windows)
│      └── Makefile          <- For generating documentation (make)
│
├── notebooks                <- Notebooks for analysis
│   ├── 001_verpla38_straatvinken_abt.ipynb                    <- exploration of abt creation
│   ├── 002_verpla38_vision_analysis.ipynb                     <- vision analysis on GSV data
│   ├── 003_verpla38_streamlit_app.py                          <- streamlit app for exploring
│   ├── 004_verpla38_modelling_baseline_elasticnet.ipynb       <- modelling exploration with elasticnet
│   ├── 004b_verpla38_modelling_gradient_boosted_trees.ipynb   <- modelling with gradient boosted trees
│   ├── 004c_verpla38_modelling_svr_ard.ipynb                  <- modelling with svr and ard models
│   ├── 005_verpla38_iteration_3.ipynb                         <- another modelling iteration
│   ├── 005_verpla38_iteration_4.ipynb                         <- another modelling iteration
│   ├── 006_verpla38_vision_analysis_antwerp_data.ipynb        <- scraping of GSV data for Antwerp dataset
│   └── <whole_lot_of_random_files>                            <- badly administered resources 
│
└── src                      <- Code for use in this project
    └── aicityflowsstraatvinken       <- Python package
        ├── __init__.py      <- Python package initialisation
        ├── 01-create-abt.py <- Code for abt creation
        |                       Note that adding google streetview data is a manual step using
        |                       notebook 002 in deep learning environment (gpu enabled)
        ├── 02-train.py      <- train models based on dataset and train_config file 
        ├── train_config_<YEAR_VERSION>.yaml
        |                    <- train configuration files, select or create one based on
        |                       available fields in ABT (latest version should suffice)
        ├── 03-infer.py      <- perform inference using trained models with infer_config file 
        ├── infer_config_<YEAR_VERSION>.yaml
        |                    <- inference configuration files, latest one should suffice
        ├── 04-dashboard.py  <- run dashboard on output of inference
        ├── utils.py         <- many utility functions
        └── hackathon_dashboards.py    <- dashboard as presented in EDIT internal hackathon




```

# Connected repositories in the Cityflows ecosystem
- the road cutter procedure: [repository](https://github.com/imec-int/cityflows-road-cutter)
- the data model: [repository](https://github.com/imec-int/cityflows-model)