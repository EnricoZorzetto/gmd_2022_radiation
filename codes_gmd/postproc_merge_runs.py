import os
import sys
import json
import time
import pandas as pd
import numpy as np
from shutil import rmtree
import xarray as xr
from itertools import product

########################################################################################################################

# AFTER THE MONTE CARLO SIMULATION:
# READ ALL THE RESULTING NETCDF AND MERGE THOSE WITH SAME PARAMETERS

########################################################################################################################


# datadir = "/Users/ez6263/Documents/rmc_datasets/"
datadir = "/Users/ez6263/"
# datadir = "/lustre/f2/dev/Enrico.Zorzetto/dem_datasets/gmd_2021_dems"
# run_merged = "output_Laptop_test_merged" # name of the desired output (merged) simulation
# run_merged = "output_cluster_PP3D_EastAlps_merged123456" # name of the desired output (merged) simulation
run_merged = "output_cluster_PP3D_Peru_merged123456" # name of the desired output (merged) simulation
# list here all the runs to be merged
runs_to_merge = [
                    # "output_Laptop_test",
                    # "output_Laptop_test_run2"
                    # "output_cluster_PP3D_Peru_run1",
                    "output_cluster_PP3D_Peru_run2",
                    "output_cluster_PP3D_Peru_run3",
                    "output_cluster_PP3D_Peru_run4",
                    "output_cluster_PP3D_Peru_run5"
                    # "output_cluster_PP3D_EastAlps_run5",
                    # "output_cluster_PP3D_EastAlps_run6"
]

########################################################################################################################





# import pickle
# numjob = 0
# with open(os.path.join(datadir, 'output_laptop_test_run1',
#                        'dfatm_{}.pickle'.format(numjob)), 'rb') as handle:
#     dfatm = pickle.load(handle)



nruns = len(runs_to_merge)
assert nruns > 1 # There must be ay least two runs in this list




dir_3d = os.path.join('output_sim', 'output_sim_3D')
dir_pp = os.path.join('output_sim', 'output_sim_PP')

cases_pp_run1 = os.listdir( os.path.join(datadir, runs_to_merge[0], dir_pp))
cases_3d_run1 = os.listdir( os.path.join(datadir, runs_to_merge[0], dir_3d))
ncases3d = len(cases_3d_run1)
ncasespp = len(cases_pp_run1)

# runs_pp_run2 = os.listdir( os.path.join(datadir, run2, dir_pp))
# runs_3d_run2 = os.listdir( os.path.join(datadir, run2, dir_3d))
dir_pp_merged = os.path.join(datadir, run_merged, dir_pp)
dir_3d_merged = os.path.join(datadir, run_merged, dir_3d)

os.system("mkdir -p {}".format(dir_pp_merged))
os.system("mkdir -p {}".format(dir_3d_merged))

files_to_copy = ['list_sim_cases_3D.csv', 'list_sim_cases_PP.csv']
for filename_i in files_to_copy:
    os.system("cp -r {} {}".format(
        os.path.join(datadir, runs_to_merge[0], filename_i),
        os.path.join(datadir, run_merged, filename_i)
    ))


# assert len(runs_3d_run1) == len(runs_3d_run2)
# assert len(runs_pp_run1) == len(runs_pp_run2)



# loop on 3d/pp and run numbers

for js in ("3D", "PP"):
    if js == '3D':
        ncases = ncases3d
    else:
        ncases = ncasespp
# for js in ["3D"]:
    print(js)
    # for i in range(1):
    for i in range(ncases):
        print(i)
        nc1 = xr.open_dataset(  os.path.join(datadir, runs_to_merge[0], "output_sim",
                "output_sim_{}".format(js), "photonmc_output_{}.nc".format(i)))


        nx = nc1.dims['lon']
        ny = nc1.dims['lat']
        ECOUP = np.zeros((nx, ny), dtype=np.float32)
        EDIR = np.zeros((nx, ny), dtype=np.float32)
        EDIF = np.zeros((nx, ny), dtype=np.float32)
        ERDIR = np.zeros((nx, ny), dtype=np.float32)
        ERDIF = np.zeros((nx, ny), dtype=np.float32)
        etoa = 0
        eabs = 0
        esrf = 0
        nphotons = 0
        ftoanorm_sum = 0

        nc_merged = xr.Dataset(
            data_vars=dict(
                ecoup=(["lon", "lat"], ECOUP),
                erdif=(["lon", "lat"], ERDIF),
                erdir=(["lon", "lat"], ERDIR),
                edir=(["lon", "lat"], EDIR),
                edif=(["lon", "lat"], EDIF),
                elev=(["lon", "lat"], nc1['elev'].values)
            ),
            coords=dict(
                lat=(["lat"], nc1.coords['lat'].values),
                lon=(["lon"], nc1.coords['lon'].values),
                lat_meters=(["lat_meters"], nc1.coords['lat_meters'].values),
                lon_meters=(["lon_meters"], nc1.coords['lon_meters'].values),
            ),
            attrs=dict(description="Simulation parameters.",
                       cosz=nc1.attrs['cosz'],
                       phi=nc1.attrs['phi'],
                       alpha_dir=nc1.attrs['alpha_dir'],
                       alpha_dif=nc1.attrs['alpha_dif'],
                       ave_elev=nc1.attrs['ave_elev'],
                       tilted=nc1.attrs['tilted'],
                       forcepscatter=nc1.attrs['forcepscatter'],
                       pscatterf=nc1.attrs['pscatterf'],
                       aerosol=nc1.attrs['aerosol'],
                       ftoanorm=nc1.attrs['ftoanorm'],
                       nphotons=0,
                       planepar=nc1.attrs['planepar'],
                       etoa=0,
                       eabs=0,
                       esrf=0
                       ),
        )


        for ir in range(nruns):
            ncir = xr.open_dataset(  os.path.join(datadir, runs_to_merge[ir], "output_sim",
                        "output_sim_{}".format(js), "photonmc_output_{}.nc".format(i)))

            # Make sure the two simulations are consistent before summing them
            assert np.isclose(nc1.attrs['cosz'],          ncir.attrs['cosz'])
            assert np.isclose(nc1.attrs['phi'],           ncir.attrs['phi'])
            assert np.isclose(nc1.attrs['alpha_dir'],     ncir.attrs['alpha_dir'])
            assert np.isclose(nc1.attrs['alpha_dif'],     ncir.attrs['alpha_dif'])
            assert np.isclose(nc1.attrs['tilted'],        ncir.attrs['tilted'])
            assert np.isclose(nc1.attrs['planepar'],      ncir.attrs['planepar'])
            assert np.isclose(nc1.attrs['forcepscatter'], ncir.attrs['forcepscatter'])
            assert np.isclose(nc1.attrs['pscatterf'],     ncir.attrs['pscatterf'])
            assert np.isclose(nc1.attrs['ftoanorm'],     ncir.attrs['ftoanorm'])

            # sum the results of the two simulations::
            vars2sum = ['edir', 'edif', 'erdif', 'erdir', 'ecoup']
            for myvar in vars2sum:
                nc_merged[myvar].values = nc_merged[myvar].values + ncir[myvar].values

            attrs2sum = ['etoa', 'eabs', 'esrf', 'nphotons']
            for myvar in attrs2sum:
                # nc_merged.attrs[myvar] = nc1.attrs[myvar] + nc2.attrs[myvar]
                nc_merged.attrs[myvar] = nc_merged.attrs[myvar] + ncir.attrs[myvar]

        # ONCE WE ARE DONE SUMMING THE SIMULATIONS, LET's WRITE THE RESULTING NETCDF FILE
        file_merged = os.path.join(os.path.join(datadir, run_merged, "output_sim", "output_sim_{}".format(js),
                                                "photonmc_output_{}.nc".format(i)))
        nc_merged.to_netcdf(file_merged)
        nc_merged.close()
