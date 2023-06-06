CONDA_ENV := straatvinken
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(CONDA_ENV)
PYTHON := python
UNAME := $(shell uname) # to derive os, either Darwin (=mac ox) Windows_NT or Linux

#segmentation model parameters
MODEL_NAME := ade20k-resnet50dilated-ppm_deepsup
MODEL_PATH := data/model/CSAILVisionSegmentation/$(MODEL_NAME)
ENCODER := $(MODEL_NAME)/encoder_epoch_20.pth
ENCODER_PATH := $(MODEL_PATH)/ckpt/$(ENCODER)
DECODER := $(MODEL_NAME)/decoder_epoch_20.pth
DECODER_PATH := $(MODEL_PATH)/ckpt/$(DECODER)
CONFIG := $(MODEL_PATH)/config/$(MODEL_NAME).yaml

INFER_SUFFIX := 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

setup:
	echo "Installation instructions"
	echo "========================="
	echo "install azcopy manually (+ add to path) from https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10"
	ifeq ($(UNAME), Darwin)
		brew install azcopy
	endif
	ifeq ($(UNAME), Windows_NT)
		# fix this if you have windows...
	endif
	ifeq ($(UNAME), Linux)
		# fix this if you have linux...
	endif
	echo "Install conda manually from https://www.anaconda.com/products/individual"

env:
	echo "Create conda environment and install dependencies"
	conda env create -f conda_env.yml

rmenv:
	echo "Remove conda environment"
	conda env remove -n $(CONDA_ENV)

envdl:
	echo "Create conda environment and install dependencies"
	conda env create -f conda_env_dl.yml

rmenvdl:
	echo "Remove conda environment"
	conda env remove -n $(CONDA_ENV)-dl

envmin:
	echo "Create conda environment and install dependencies"
	conda env create -f conda_env_minimal.yml

rmenmin:
	echo "Remove conda environment"
	conda env remove -n $(CONDA_ENV)-minimal

download:
	echo "Downloading the data"
	
	# create directories
	mkdir -p data/raw/straatvinken
	mkdir -p data/raw/population_density_statbel
	mkdir -p data/raw/number_of_cars_statbel
	mkdir -p data/raw/traffic_accidents_statbel
	mkdir -p data/raw/google_streetview
	mkdir -p data/processed/google_streetview_mask
	mkdir -p data/processed/google_streetview_masked
	mkdir -p data/external/CSAILVision

	# download statistical sectors
	wget -a data/raw/dl.log -O "data/raw/sh_statbel_statistical_sectors_31370_20210101.shp.zip" "https://statbel.fgov.be/sites/default/files/files/opendata/Statistische%20sectoren/sh_statbel_statistical_sectors_31370_20210101.shp.zip"
	# download building density
	wget -a data/raw/dl.log -O "data/raw/lu_bebdicht_ha_vlaa_2019_v2.zip" "https://www.milieuinfo.be/dms/d/d/workspace/SpacesStore/386c0e41-3c77-4a30-b81a-75db3ef72c14/lu_bebdicht_ha_vlaa_2019_v2.zip"
	# download population density
	wget -a data/raw/dl.log -O "data/raw/population_density_statbel/OPENDATA_SECTOREN_2021.xlsx" "https://statbel.fgov.be/sites/default/files/files/opendata/bevolking/sectoren/OPENDATA_SECTOREN_2021.xlsx"
	# download cars and households
	wget -a data/raw/dl.log -O "data/raw/number_of_cars_statbel/TF_CAR_HH_SECTOR.xlsx" "https://statbel.fgov.be/sites/default/files/files/opendata/Aantal%20wagens%20per%20statistische%20sector/TF_CAR_HH_SECTOR.xlsx"
	# download wegenregister
	wget -a data/raw/dl.log -O "data/raw/Wegenregister_SHAPE_20220317.zip" "https://downloadagiv.blob.core.windows.net/wegenregister/Wegenregister_SHAPE_20220317.zip"
	# download population grid
	wget -a data/raw/dl.log -O "data/raw/TF_POPULATION_GRID_3035_20200101_shp.zip" "https://statbel.fgov.be/sites/default/files/files/opendata/Pop_GRID/TF_POPULATION_GRID_3035_20200101_shp.zip"
	# download traffic accidents
	wget -a data/raw/dl.log -O "data/raw/traffic_accidents_statbel/TF_ACCIDENTS_2020.xlsx" "https://statbel.fgov.be/sites/default/files/files/opendata/Verkeersongevallen/TF_ACCIDENTS_2020.xlsx"
	# wget -a data/raw/dl.log -O "data/raw/traffic_accidents_statbel/TF_ACCIDENTS_VICTIMS_2020.xlsx" "https://statbel.fgov.be/sites/default/files/files/opendata/Verkeersslachtoffers/TF_ACCIDENTS_VICTIMS_2020.xlsx"
	# download ruimterapport shapefiles
	wget -a data/raw/dl.log -O "data/raw/lu_klv_vlaa_2019.zip" "https://www.milieuinfo.be/dms/d/d/workspace/SpacesStore/c55061af-8ae2-4be1-aeb2-83b17441eb47/lu_klv_vlaa_2019.zip"
	# download school shapefiles -> this will not work, as AIV doesn't allow direct downloads
	wget -a data/raw/dl.log -O "data/raw/Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service_Shapefile.zip" "https://download.vlaanderen.be/Bestellingen/Download/657538"
	
	wget -a data/raw/dl.log -O "data/raw/lu_landgebruik_vlaa_2019_v2.zip" "https://www.milieuinfo.be/dms/d/d/workspace/SpacesStore/95978609-c38b-4f51-8d61-5e893242feef/lu_landgebruik_vlaa_2019_v2.zip"
	# OSM data
	wget -a data/raw/dl.log -O "data/raw/osm-belgium-latest-free.shp.zip" "https://download.geofabrik.de/europe/belgium-latest-free.shp.zip"


	# model checkpoints & config
	if [ ! -e $(MODEL_PATH) ]; then mkdir -p $(MODEL_PATH); fi
	if [ ! -e $(MODEL_PATH)/ckpt ]; then mkdir -p $(MODEL_PATH)/ckpt; fi
	if [ ! -e $(MODEL_PATH)/config ]; then mkdir -p $(MODEL_PATH)/config; fi
	if [ ! -e $(ENCODER_PATH) ]; then wget -P $(MODEL_PATH)/ckpt http://sceneparsing.csail.mit.edu/model/pytorch/$(ENCODER); fi
	if [ ! -e $(DECODER_PATH) ]; then wget -P $(MODEL_PATH)/ckpt http://sceneparsing.csail.mit.edu/model/pytorch/$(DECODER); fi
	wget -a data/raw/dl.log -O $(CONFIG) "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/config/ade20k-resnet50dilated-ppm_deepsup.yaml"
	# download color matrix
	wget -a data/raw/dl.log -O "data/external/CSAILVision/color150.mat" "https://github.com/CSAILVision/sceneparsing/raw/master/visualizationCode/color150.mat"
	# download object categories
	wget -a data/raw/dl.log -O "data/external/CSAILVision/object150_info.csv" "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/object150_info.csv"
	
	# the next statement will probably not run
	azcopy copy "https://cityflowsdev.blob.core.windows.net/data-files/mobiele_stad/validation_straatvinken/transformed/*?sp=r&st=2022-01-12T16:31:13Z&se=2022-01-13T00:31:13Z&spr=https&sv=2020-08-04&sr=b&sig=flG1va%2FnJ09o0f9UcbHeaTreFXcjwPFJjLl6G0id3qs%3D" "data/raw/straatvinken/SV2020_DataVVR-Antwerp_20210422_transformed_enriched.csv"

extract:
	echo "Extracting the data"
	unzip -o data/raw/sh_statbel_statistical_sectors_31370_20210101.shp.zip -d data/raw
	unzip -o data/raw/lu_bebdicht_ha_vlaa_2019_v2.zip -d data/raw/lu_bebdicht_ha_vlaa_2019_v2
	unzip -o data/raw/Wegenregister_SHAPE_20220317.zip -d data/raw
	unzip -o data/raw/TF_POPULATION_GRID_3035_20200101_shp.zip -d data/raw
	unzip -o data/raw/lu_klv_vlaa_2019.zip -d data/raw
	unzip -o data/raw/Onderwijsaanbod_in_Vlaanderen_en_Brussel_via_POI_service_Shapefile.zip -d data/raw
	unzip -o data/raw/lu_landgebruik_vlaa_2019_v2.zip -d data/raw
	unzip -o data/raw/osm-belgium-latest-free.shp.zip -d data/raw/

abt:
	$(CONDA_ACTIVATE) && $(PYTHON) src/aicityflowsstraatvinken/01-create-abt.py

train:
	$(CONDA_ACTIVATE) && $(PYTHON) src/aicityflowsstraatvinken/02-train.py

infer:
	$(foreach var,$(INFER_SUFFIXES),$(CONDA_ACTIVATE) && $(PYTHON) src/aicityflowsstraatvinken/03-infer.py --suffix $(var);)

data: download extract abt

all: env data train infer

demo:
	$(CONDA_ACTIVATE)-minimal && streamlit run src/aicityflowsstraatvinken/04-dashboard.py