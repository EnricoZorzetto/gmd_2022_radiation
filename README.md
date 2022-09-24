# Accounting for Radiation-topography interactions in the GFDL land model version 4.2

This repository includes the code used for analyzing the sub--grid distribution of solar radiation over mountains in the paper "Effects of complex terrain on the shortwave radiative balance : A sub--grid scale parameterization for the GFDL Land Model version 4.2" by E. Zorzetto et al. The python code processes terrain spatial data and fields of solar radiation over complex topography and produces the results shown in the manuscript, performing the training and testing of the predictive model for solar radiation over complex topography.

Input data used in the analysis can be access at the following Zenodo repository with DOI

```
10.5281/zenodo.6975857
```
This repository includes
- Fields of solar irradiance over flat and 3D terrain for different solar angles
- Terrain information for the three domains in the study, which include digital elevation maps, derived topographic variables (sky view, terrain configuration, and slope and aspect --derived quantities) and a partition of the domain in tiles for different tiling configurations. Different tiling configuraions are stored with a naming convention which identifies the number and type opf subgrid units used.

Terrain maps for the three domains in the study are stored with the following naming convention, which is based on the domain partitioning used to create tiles. For example, the file 
```
gmd_2021_grids_light_k5n1pV.zip
```
refers to the domain terrain information relative to the case of grid cell subdivision in k=5 hillslopes, a single height band (n=1), and a variable number of land sub-units (variable p, or pV). The content of this file will correspond to that of the input directory (see workspace description below):
```
<simul_folder>/gmd_2021/gmd_2021_data/gmd_2021_grids_light/k5n1pV
``` 

# Installation


The computing environment (used in Mac OS system) can be built using Conda and the yaml file env_rad_macos.yml

```
conda env create env_rad_macos.yml
conda activate rad
```


# Running the analysis

The analysis in the paper can be replicated by running 
 

```
python codes_gmd/main_run_gmd_2021_combined.py
```


# Setting up the workspace

The code will need to point to a user-specified directory <simul_folder> containing input and output model data. The following input must be provided

    - <simul_folder>/gmd_2021/gmd_2021_data/gmd_2021_grids_light/k5n1pV
    - <simul_folder>/gmd_2021/gmd_2021_data/gmd_2021_grids_light/kVn1p5
    - <simul_folder>/gmd_2021/gmd_2021_data/output_cluster_PP3D_EastAlps_merged123456
    - <simul_folder>/gmd_2021/gmd_2021_data/output_cluster_PP3D_Peru_merged123456

with corresponding directory stored in the data repository.
A directory with the results will be created in <simul_folder>/gmd_2021/gmd_2021_output/

Where the directory "simul_folder" can be specified by the user in the namelist in python codes_gmd/main_run_gmd_2021_combined.py
Code used to construct the domain (stored in gmd_2021_grids_light) and to simulate the radiation fields (stored in output_cluster_PP3D_*) is included in GFDL_preprocessing and rmc directories, respectively.

# land grid data

The grid data (maps of terrain parameters and spatial classification of land in clusters) is stored in
```
<simul_folder>/gmd_2021/gmd_2021_data/gmd_2021_grids_light/
``` 
For different tiling configuration. This dataset was obtained running the preprocessing code in GFDL_preprocessing/

# monte carlo radiation fields

The software package used for performing ray tracing simulations is included in the rmc/ folder. Input data used for rmc simulations are included in the zenodo repository, and consist of terrain elevation data  (in the file GFDL_preproc_dems) and atmospheric optical properties (in single-time-step)

