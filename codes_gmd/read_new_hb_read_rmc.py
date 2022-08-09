import os
import re
import struct
import numpy as np
import pandas as pd
import h5py
import netCDF4
import xarray as xr
import geopy.distance
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# import horizon as hor
from topocalc.viewf import viewf, gradient_d8
import pickle
from geospatialtools import gdal_tools
# import tensorflow as tf
import demfuncs as dem
import tilefuncs as til
import string
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

import matplotlib
# matplotlib.use('Agg')
matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()

output_dir = os.path.join('..', '..', '..', 'Documents', 'temp_outputfig')
simul_name = 'output_cluster_PP3D_EastAlps_run1_a35'
simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP

print(os.listdir(simul_folder))
datadir = os.path.join(simul_folder, simul_name)

df3d = pd.read_csv(os.path.join(datadir, 'list_sim_cases_3D.csv'))
# dfpp = pd.read_csv( os.path.join(datadir, 'list_sim_cases_PP.csv'))

COSZs = list(np.unique(df3d['cosz']))
PHIs = list(np.unique(df3d['phi']))
ADIRs = list(np.unique(df3d['adir']))
# print('cosz available = {}'.format(COSZs))
# print('phi  available = {}'.format(PHIs))
# print('adir available = {}'.format(ADIRs))

# fluxesc = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']

nazimuths = np.size(PHIs)
nadir = np.size(ADIRs)
ncosz = len(COSZs)

crop_buffer = 0.20; do_average = False; aveblock = 100; FTOAnorm = 1362.0
mycosz = 0.3; myphi = 0.0; myadir = 0.5

res = dem.read_PP_3D_differences(datadir=datadir,
                           outdir=output_dir,
                           do_average=do_average,
                           aveblock=aveblock,
                           crop_buffer=crop_buffer,
                           FTOAnorm=FTOAnorm)

rest_ip = dem.load_terrain_vars(cosz=mycosz, phi=myphi,
                            buffer=crop_buffer,
                            do_average=do_average,
                            aveblock=aveblock,
                            datadir=datadir)

res3d_ip = dem.load_3d_fluxes(FTOAnorm=FTOAnorm,
                          cosz=mycosz,
                          phi=myphi,
                          adir=myadir,
                          do_average=do_average,
                          aveblock=aveblock,
                          buffer=crop_buffer,
                          datadir=datadir)

respp_ip = dem.load_pp_fluxes(cosz=mycosz,
                          adir=myadir,
                          FTOAnorm=FTOAnorm,
                          do_average=False, aveblock=1,
                          buffer=crop_buffer,
                          datadir=datadir)

sia_field = np.ravel(rest_ip['SIAnorm'])
tcf_field = np.ravel(rest_ip['TCFnorm'])
svf_field = np.ravel(rest_ip['SVFnorm'])
fFMdir_field = (res3d_ip['FMdir'] - respp_ip['Fdir_pp']) / respp_ip['Fdir_pp']
fFMdif_field = (res3d_ip['FMdif'] - respp_ip['Fdif_pp']) / respp_ip['Fdif_pp']
fFMrdir_field = (res3d_ip['FMrdir']) / respp_ip['Fdir_pp']
fFMrdif_field = (res3d_ip['FMrdif']) / respp_ip['Fdif_pp']
fFMcoup_field = (res3d_ip['FMcoup'] - respp_ip['Fcoup_pp']) / respp_ip['Fcoup_pp']

sia = np.ravel( sia_field )
tcf = np.ravel( tcf_field )
svf = np.ravel( svf_field )
fFMdir  = np.ravel( fFMdir_field )
fFMdif  = np.ravel( fFMdif_field )
fFMrdir = np.ravel( fFMrdir_field )
fFMrdif = np.ravel( fFMrdif_field )
fFMcoup = np.ravel( fFMcoup_field )

plt.figure()
plt.plot(sia, fFMdir, 'o')
plt.show()



plt.figure(figsize=(6,6))
plt.imshow(fFMdir_field)
# plt.xlabel('sia'); plt.ylabel('fdir')
plt.savefig( os.path.join(output_dir, 'fdir_map.png'))
plt.close()


plt.figure(figsize=(6,6))
plt.plot(sia, fFMdir, 'o')
plt.xlabel('sia'); plt.ylabel('fdir')
plt.savefig( os.path.join(output_dir, 'sia_vs_dir.png'))
plt.close()


plt.figure(figsize=(6,6))
plt.plot(svf, fFMdif, 'o')
plt.xlabel('svf'); plt.ylabel('fdif')
plt.savefig( os.path.join(output_dir, 'svf_vs_dif.png'))
plt.close()

plt.figure(figsize=(6,6))
plt.plot(tcf, fFMdif, 'o')
plt.xlabel('tcf'); plt.ylabel('fdif')
plt.savefig( os.path.join(output_dir, 'tcf_vs_dif.png'))
plt.close()
