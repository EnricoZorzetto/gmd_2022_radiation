# Accounting for Radiation-topography interactions in the GFDL lan model

Codes used for analyzing the sub--grid distribution of solar radiation over mountains in the paper "Effects of complex terrain on the shortwave radiative balance : A sub--grid scale parameterization for the GFDL Land Model version 4.2" by E. Zorzetto et al.

Input data can be access at the following Zenodo repository 10.5281/zenodo.6975857, which include
- Fields of solar irradiance over flat and 3D terrain for different solar angles
- Terrain information for the three domains in the study, which include digital elevation maps, derived topographic variables (sky view, terrain configurations, slope and aspect derived quantities) and partition of the domain in tiles for different tiling configuraions 


# Installation


The computing environment can be built using Conda and the yaml file env_rad_macos.yml


# Running the analysis

The analysis in the paper can be replicated using the script
 

codes_gmd/main_run_gmd_2021_combined.py


The code will need to point to a user-specified directory USERDIR with the data:

    - $USERDIR/gmd_2021/gmd_2021_data/gmd_2021_grids_light
    - $USERDIR/gmd_2021/gmd_2021_data/output_cluster_PP3D_EastAlps_merged123456
    - $USERDIR/gmd_2021/gmd_2021_data/output_cluster_PP3D_Peru_merged123456

A directory with the results will be created in $USERDIR/gmd_2021/gmd_2021_output/

Code used to construct the domain (stored in gmd_2021_grids_light) and to simulate the radiation fields (stored in output_cluster_PP3D_*) is included in GFDL_preprocessing and rmc directories, respectively.


