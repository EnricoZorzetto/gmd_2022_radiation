
import os
import pickle
import sys
import json
import time

import pandas as pd
import xarray as xr
import numpy as np
import photon_mc_numba_fractional as ph
import matplotlib.pyplot as plt

import photon_mc_create_list_of_jobs as io
# import photon_mc_land as land
# import photon_mc_atmosphere as atm
import photon_mc_numba_fractional as ph
import photon_mc_merge_resulting_netcdfs as merge
import main_photon_cluster as photomc
import multiprocessing
import photon_mc_land as land

"""
Same workflow as done in the cluster (slurm) 
"""


# def main():

if len(sys.argv) < 2:  # if no experiment file provided, use the default
    mdfile = os.path.join('exp', 'laptop.json')
    # mdfile = os.path.join('exp', 'laptop_coarse.json')
else:  # run as::$ python program.py exp/experiment.json
    mdfile = sys.argv[1]
# mdfile = os.path.join('exp', 'laptop.json')



print('mdfile = {}'.format(mdfile))
metadata = json.load(open(mdfile, 'r'))
print('metadata = ', metadata)
numjobs = io.init_simulation_wrapper(metadata)

exp_name = metadata['name']
datadir = metadata['datadir']
outdir = os.path.join(datadir, 'output_{}'.format(exp_name))
tempdir = os.path.join(outdir, 'output_temp')
outfigdir = os.path.join(outdir, 'outfigdir')

if not os.path.exists(tempdir):
    os.makedirs(tempdir)
if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

# NEW: EA DEM, must do preprocessing here

# land.preprocess_land_dem(metadata, periodic_buffer = 10, crop = 0.25, eares = 90)
# land.preprocess_land_dem(metadata, periodic_buffer = 10, crop = 0.25, eares = 90)
land.preprocess_land_dem(metadata)

job_index = 0
photomc.run_photomc(job_index, metadata=metadata)
# print('mid time = {} minutes'.format((time.time() - init_time)/60.0))
# photomc.run_photomc(job_index, metadata=metadata) # second time, does it recompile???
simuldata = pd.read_csv( os.path.join(outdir, 'list_jobs_params.csv'))
mysim = simuldata.iloc[job_index]

# read and plot results:
# init_time = time.time()
ds = xr.open_dataset( os.path.join(tempdir, 'photonmc_output_temp_{}.nc'.format(job_index)))

# print('****************************************************')
# print('read time = {} minutes'.format((time.time() - init_time)/60.0))
# print('****************************************************')

print(list(ds.variables))
print(list(ds.attrs))

y = ds['lat']
x = ds['lon']
Z = ds['elev']

eTOA = ds.attrs['etoa']
eSRF = ds.attrs['esrf']
eABS = ds.attrs['eabs']

ecoup = ds['ecoup'].values
edir  = ds['edir'].values
edif  = ds['edif'].values
erdir = ds['erdir'].values
erdif = ds['erdif'].values
etot  = edir + edif + erdir + erdif + ecoup


print('sum ECOUP = ', np.sum(ecoup))
print('sum EDIR = ',  np.sum(edir))
print('sum EDIF = ',  np.sum(edif))
print('sum REDIR = ', np.sum(erdir))
print('sum REDIF = ', np.sum(erdif))
print('etot = ', np.sum(etot))
print('****************************************************')
print('photons escaped = ', eTOA)
print('photons absorbed = ', eABS)
print('photons impacted = ', eSRF)

# print('Profiling memory usage: [for linux systems Only]')
# mem = ph.memory_usage()
# print(mem)
# print('****************************************************')

Y, X = np.meshgrid(y, x)


# EDIR2 = (edir).astype(float)
# EDIR2[EDIR2 < 1E-6] = np.nan

edir2 = edir.copy()
edir2[edir2 < 1E-6] = np.nan

ph.matplotlib_update_settings()
plotlast = False
if plotlast:

    # plt.figure()
    # plt.imshow(Z)
    # plt.savefig(os.path.join(outfigdir, 'dem_imshow.png'), dpi = 300)
    # plt.show()
    plt.figure()
    plt.pcolormesh(X, Y, Z, alpha = 0.3, shading='auto')
    plt.pcolormesh(X, Y, edir2, cmap='jet', shading='auto')
    # plt.savefig(os.path.join(outfigdir, 'dem_pcolormesh.png'), dpi = 300)
    plt.savefig(os.path.join(outfigdir, 'impacts.png'), dpi = 300)
    plt.show()
    # plt.figure()
    # plt.imshow(Z, alpha = 0.6, extent=(x[0], x[-1], y[-1], y[0]))
    # plt.imshow(Z, alpha = 0.6)
    # plt.pcolormesh(y, x, Z, alpha = 0.6, shading='flat')
    # cbar = plt.colorbar()
    # cbar.set_label('Elev. [m msl]')
    # cbar.set_label('Elev. [m msl]')
    # plt.xlabel('x EAST [DEM grid points]')
    # plt.ylabel('y NORTH [DEM grid points]')
    # plt.imshow(edir2, cmap='jet')
    # plt.pcolormesh(y, x, EDIR2[:nby, :nbx])
    # plt.pcolormesh(y, x, edir2)
    # plt.title('cosz = {}, $\phi$ = {}, $\\alpha$ = {}, '
    #           'noscatter'.format(-cosz, phi, alpha_dir))
    # plt.gca().invert_yaxis()
    # plt.savefig(os.path.join(outfigdir, 'impacts.png'), dpi = 300)
    # plt.show()

    pickled_dem = os.path.join(outdir, 'preprocessed_dems',
                               '{}_dem_input.pkl'.format(metadata['domain']))
    with open(pickled_dem, "rb") as input_file:
        demdict = pickle.load(input_file)

    print(demdict.keys())
    lat = demdict['lats']
    lon = demdict['lons']
    Z = demdict['Z']
    LAT, LON = np.meshgrid(lat, lon)

    if plotlast:
        plt.figure(figsize=(6,6))
        plt.pcolormesh(LON, LAT, Z, shading = 'auto')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.tight_layout()
        plt.savefig(os.path.join(outfigdir, 'dem.png'), dpi = 300)
        plt.show()


# if __name__ == '__main__':
#     main()