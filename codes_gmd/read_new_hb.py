import os
import re
import struct
import numpy as np
import pandas as pd
import h5py
import netCDF4
import xarray as xr
# import geopy.distance
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

# matplotlib.use('Agg')
matplotlib.use('Qt5Agg') # dyn show plots

plt.figure()
plt.plot([1, 2, 3], [1, 2, 3], 'o')
plt.show()

dem.matplotlib_update_settings()

output_dir = os.path.join('..', '..', '..', 'Documents', 'temp_outputfig')
if not os.path.exists(output_dir): os.makedirs(output_dir)

datadir_all = os.path.join('..', '..', '..', 'Documents', 'gmd_2021_grids_light')

ftilename = 'res_EastAlps_k_20_n_1_p_2'
# ftilename = 'res_EastAlps_k_2_n_1_p_2'
# ftilename = 'res_Peru_k_2_n_1_p_2'
datadir = os.path.join(datadir_all, 'kVn1p2', ftilename)
landdir = os.path.join(datadir, 'land', 'tile:1,is:1,js:1')

os.listdir(datadir)


dbfile = os.path.join(datadir, 'ptiles.{}.tile1.h5'.format(ftilename))
dbs = h5py.File(dbfile, 'r')
print(dbs.keys())
print(dbs['grid_data']['tile:1,is:1,js:1'].keys())
print(dbs['grid_data']['tile:1,is:1,js:1']['metadata'].keys())
tid = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['tid'][:]
frac = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['frac'][:]
print(frac)
assert np.isclose( np.sum(frac), 1.0)
tile = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['tile'][:]
ntiles = len(tile)
ttype = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['type'][:]

# gel glacier values
# gl = dbs['grid_data']['tile:1,is:1,js:1']['glacier'].keys()
# glF = dbs['grid_data']['tile:1,is:1,js:1']['glacier']['frac'][:]
# glSC = dbs['grid_data']['tile:1,is:1,js:1']['glacier']['sinscosa'][:]
# glSS = dbs['grid_data']['tile:1,is:1,js:1']['glacier']['sinssina'][:]
# glSV = dbs['grid_data']['tile:1,is:1,js:1']['glacier']['svf'][:]
# glTC = dbs['grid_data']['tile:1,is:1,js:1']['glacier']['tcf'][:]
la = dbs['grid_data']['tile:1,is:1,js:1']['lake'].keys()
laF = dbs['grid_data']['tile:1,is:1,js:1']['lake']['frac'][:]
laSC = dbs['grid_data']['tile:1,is:1,js:1']['lake']['sinscosa'][:]
laSS = dbs['grid_data']['tile:1,is:1,js:1']['lake']['sinssina'][:]
laSV = dbs['grid_data']['tile:1,is:1,js:1']['lake']['svf'][:]
laTC = dbs['grid_data']['tile:1,is:1,js:1']['lake']['tcf'][:]
soil = dbs['grid_data']['tile:1,is:1,js:1'][  'soil'].keys()
soilF = dbs['grid_data']['tile:1,is:1,js:1'][ 'soil']['frac'][:]
# np.sum(soilF)
soilSC = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_sinscosa'][:]
soilSS = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_sinssina'][:]
soilSV = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_svf'][:]
soilTC = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_tcf'][:]


mycosz = 0.3
myazi = 0.0*np.pi
myalbedo = 0.3

dbs_tvr_vars = ['sinssina', 'sinscosa', 'svf', 'tcf']
dbs_trv = { elem : np.zeros(ntiles) for elem in dbs_tvr_vars}
for elem in dbs_tvr_vars:
    # print(elem)
    # elemn = elem
    # if elem == 'svf': elemn = 'svfn'
    # if elem == 'tcf': elemn = 'tcfn'
    if 'glacier' in list(dbs['grid_data']['tile:1,is:1,js:1'].keys()):
        dbs_trv[elem][0] = dbs['grid_data']['tile:1,is:1,js:1']['glacier'][elem][:]
        dbs_trv[elem][1] = dbs['grid_data']['tile:1,is:1,js:1']['lake'][elem][:]
        dbs_trv[elem][2:] = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_{}'.format(elem)][:]
    else:
        dbs_trv[elem][0] = dbs['grid_data']['tile:1,is:1,js:1']['lake'][elem][:]
        dbs_trv[elem][1:] = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_{}'.format(elem)][:]
# dbs_trv['sian'] = 1 + np.sqrt(1-mycosz**2)/mycosz * ( np.cos(myazi)*dbs_trv['sinscosa'] + np.sin(myazi)*dbs_trv['sinssina'])
dbs_trv['sian'] = mycosz + np.sqrt(1-mycosz**2) * ( np.cos(myazi)*dbs_trv['sinscosa'] + np.sin(myazi)*dbs_trv['sinssina'])
dbs_trv['tcfn'] = dbs_trv.pop('tcf'); dbs_trv['svfn'] = dbs_trv.pop('svf')

# TILE_FRACS = np.zeros(ntiles)

# 695         solar_inc = 1.0 + (sinz/cosz) * (cos(azi_angle) * sinsl_cosas + &
# 696                                          sin(azi_angle) * sinsl_sinas)





# LOAD HIGH RES MAPS
# tiles = np.ravel( gdal_tools.read_data( os.path.join( landdir, 'tiles.tif')).data  )
hrSC =  np.ravel( gdal_tools.read_data( os.path.join( landdir, 'sinscosa_ea.tif')).data  )
hrSS =  np.ravel( gdal_tools.read_data( os.path.join( landdir, 'sinssina_ea.tif')).data  )
hrSV =  np.ravel( gdal_tools.read_data( os.path.join( landdir, 'svf_ea.tif')).data  )
hrTC =  np.ravel( gdal_tools.read_data( os.path.join( landdir, 'tcf_ea.tif')).data  )
# hrSIA = 1 + np.sqrt(1-mycosz**2)/mycosz * ( np.cos(myazi)*hrSC + np.sin(myazi)*hrSS )
hrSIA = mycosz + np.sqrt(1-mycosz**2) * ( np.cos(myazi)*hrSC + np.sin(myazi)*hrSS )
hrs_trv = {'sian':hrSIA, 'svfn':hrSV, 'tcfn':hrTC, 'sinssina':hrSS, 'sinscosa':hrSC}

ave_trv = {key: np.mean(hrs_trv[key]) for key in hrs_trv.keys()}
avedb_trv = {key: np.sum( frac * dbs_trv[key]) for key in dbs_trv.keys()}

dfp = {'sian':mycosz, 'svfn':1, 'tcfn':0}

# UCLA DATASET
ucla_sky_view = 0.981315    
ucla_terrain_config = 0.812428E-01
ucla_sinsl_cosas = 0.619293E-02
ucla_sinsl_sinas =-0.127051E-01
ucla_sia = mycosz + np.sqrt(1-mycosz**2) * ( np.cos(myazi)*ucla_sinsl_cosas + 
                                             np.sin(myazi)*ucla_sinsl_sinas )
ucla_trv = {'sian':ucla_sia, 'svfn':ucla_sky_view, 'tcfn':ucla_terrain_config}


flux_vars = ['fdir', 'frdir', 'fdif', 'frdif']
for fv in flux_vars:
    dbs_trv[fv] = dem.lee_model_predict(dbs_trv, label=fv, cosz=mycosz, albedo=myalbedo)
    hrs_trv[fv] = dem.lee_model_predict(hrs_trv, label=fv, cosz=mycosz, albedo=myalbedo)
    ave_trv[fv] =  dem.lee_model_predict(ave_trv, label=fv, cosz=mycosz, albedo=myalbedo)
    avedb_trv[fv] =dem.lee_model_predict(avedb_trv, label=fv, cosz=mycosz, albedo=myalbedo)
    ucla_trv[fv] =dem.lee_model_predict(ucla_trv, label=fv, cosz=mycosz, albedo=myalbedo)
    
    dfp[fv] =  dem.lee_model_predict(dfp, label=fv, cosz=mycosz, albedo=myalbedo)

dbs_ave = {}; hrs_ave = {}
mykeys = ['fdir', 'frdir', 'fdif', 'frdif', 'sian', 'sinssina', 'sinscosa', 'svfn', 'tcfn']
for elem in mykeys:
    dbs_ave[elem] = np.sum(frac * dbs_trv[elem])
    hrs_ave[elem] = np.mean(hrs_trv[elem])
    print('{:s} average: tiles =: {:.5f}, high-res =: {:.5f}'.format(elem, dbs_ave[elem], hrs_ave[elem]))


# np.sum(frac*dbs_trv['fdir'])

np.min(hrs_trv['fdir'])
np.max(hrs_trv['fdir'])
np.mean(hrs_trv['fdir'])
np.median(hrs_trv['fdir'])
np.std(hrs_trv['fdir'])


# np.mean(hrs_trv['svfn'])
# np.mean(rmc_trv['svfn'])

plt.figure()
plt.plot(hrs_trv['sian'], hrs_trv['fdir'], 'o')
# plt.plot(hrs_trv['svfn'], hrs_trv['fdif'], 'o')
plt.savefig( os.path.join(output_dir, 'sia_vs_fdir_hr.png'))
plt.show()



# COMPUTE AND COMPARE GRID CELL AVERAGE VALUES
# ^^^^^^^^^^^^^^^^^

# &&&&&&&&&&&&&&&&&

# COMPUTE RAD CORRECTIONS USING WLL


# plt.figure(figsize=(6,6))
# upperl = np.mean(svf.data) + 3*np.std(svf.data)
# lowerl = np.mean(svf.data) - 3*np.std(svf.data)
# plt.imshow(svf.data, extent = [svf.minx, svf.maxx, svf.miny, svf.maxy],
#                     vmin=lowerl, vmax=upperl)
# plt.colorbar()
# plt.savefig( os.path.join(output_dir, 'svf_map.png'))
# plt.close()

# utiles = np.unique(tiles.data)
# ntiles = np.size(utiles) - 1
# print('ntiles = {}'.format(ntiles))
# plt.figure(figsize=(6,6))
# # upperl = np.mean(tiles.data) + 3*np.std(tiles.data)
# # lowerl = np.mean(tiles.data) - 3*np.std(tiles.data)
# tiles.data[tiles.data<0] = np.nan
# bounds=np.arange(ntiles)
# cmap = 'Pastel1'
# from matplotlib import colors
# # norm = colors.BoundaryNorm(bounds, cmap.N)
# plt.imshow(tiles.data,
#            extent = [tiles.minx, tiles.maxx, tiles.miny, tiles.maxy],
#            cmap=cmap)
#            # vmin=lowerl, vmax=upperl)
# plt.colorbar()
# plt.savefig( os.path.join(output_dir, 'tiles_map.png'))
# plt.close()



# compare with RMC results for the same solar angles
# FTOAnorm=1361; myadir = 0.3; do_average = True; aveblock = 6; crop_buffer = 0.1
FTOAnorm=1300; myadir = 0.3; do_average = False; aveblock = 4; crop_buffer = 0.37

# output_dir = os.path.join('..', '..', '..', 'Documents', 'temp_outputfig')
simul_name = 'output_cluster_PP3D_EastAlps_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run1_a35'
simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP
datadir_rmc = os.path.join(simul_folder, simul_name)
res3d_ip = dem.load_3d_fluxes(FTOAnorm=FTOAnorm,
                              cosz=mycosz,
                              phi=myazi,
                              adir=myadir,
                              do_average=do_average,
                              aveblock=aveblock,
                              buffer=crop_buffer,
                              datadir=datadir_rmc)

respp_ip = dem.load_pp_fluxes(cosz=mycosz,
                              adir=myadir,
                              FTOAnorm=FTOAnorm,
                              do_average=False,
                              aveblock=1,
                              buffer=crop_buffer,
                              datadir=datadir_rmc)

rest_ip = dem.load_terrain_vars(cosz=mycosz, phi=myazi,
                                buffer=crop_buffer,
                                do_average=do_average,
                                aveblock=aveblock,
                                datadir=datadir_rmc)

fFMdir_field = (res3d_ip['FMdir'] - respp_ip['Fdir_pp']) / respp_ip['Fdir_pp']
fFMdif_field = (res3d_ip['FMdif'] - respp_ip['Fdif_pp']) / respp_ip['Fdif_pp']
fFMrdir_field = (res3d_ip['FMrdir']) / respp_ip['Fdir_pp']
fFMrdif_field = (res3d_ip['FMrdif']) / respp_ip['Fdif_pp']
fFMcoup_field = (res3d_ip['FMcoup'] - respp_ip['Fcoup_pp']) / respp_ip['Fcoup_pp']

rmc_res = {'fdir':np.ravel(fFMdir_field),
           'fdif':np.ravel(fFMdif_field),
           'frdir':np.ravel(fFMrdir_field),
           'frdif':np.ravel(fFMrdif_field),
            }

# sian_field = rest_ip['SIAnorm']

rmc_trv = {'sian': np.ravel(rest_ip['SIAnorm']),
           'svfn': np.ravel(rest_ip['SVFnorm']),
           'tcfn': np.ravel(rest_ip['TCFnorm']),
           }
for fv in flux_vars:
    rmc_trv[fv] = dem.lee_model_predict(rmc_trv, label=fv, cosz=mycosz, albedo=myalbedo)


np.max(fFMdir_field)
# np.min(fFMdir_field)
np.min(hrs_trv['fdir'])
np.mean(fFMdir_field)
np.median(fFMdir_field)


np.mean(hrs_trv['svfn'])


print('rmc, fdir', np.mean(fFMdir_field))
print('rmc, fdif', np.mean(fFMdif_field))
print('rmc, frdir', np.mean(fFMrdir_field))
print('rmc, frdif', np.mean(fFMrdif_field))

print('hrt, fdir',  hrs_ave['fdir'])
print('hrt, fdif',  hrs_ave['fdif'])
print('hrt, frdir', hrs_ave['frdir'])
print('hrt, frdif', hrs_ave['frdif'])

print('dbs, fdir',  dbs_ave['fdir'])
print('dbs, fdif',  dbs_ave['fdif'])
print('dbs, frdir', dbs_ave['frdir'])
print('dbs, frdif', dbs_ave['frdif'])

print('aver tew, fdir',  ave_trv['fdir']) # est based on ave hr terrain
print('aver tew, fdif',  ave_trv['fdif'])
print('aver tew, frdir', ave_trv['frdir'])
print('aver tew, frdif', ave_trv['frdif'])

print('aver dbs, fdir',  avedb_trv['fdir']) # est based on ave over tiled terrain
print('aver dbs, fdif',  avedb_trv['fdif'])
print('aver dbs, frdir', avedb_trv['frdir'])
print('aver dbs, frdif', avedb_trv['frdif'])

rmc_fdir = np.mean(fFMdir_field)
rmc_fdir_stdv = np.std(fFMdir_field)

print('RMC fdir average = {}; stdv = {}'.format(rmc_fdir, rmc_fdir_stdv))


hrs_fdir = np.mean(hrs_trv['fdir'])
hrs_fdir_stdv = np.std(hrs_trv['fdir'])
dbs_fdir = np.mean(dbs_ave['fdir'])
#
print('HRS fdir average = {}; stdv = {}'.format(hrs_fdir, hrs_fdir_stdv))

print('BDS fdir average = {}'.format(dbs_fdir))

plt.figure()
# plt.plot(hrs_trv['sian'], hrs_trv['fdir'], 'o')
plt.hist( np.ravel(fFMdir_field), bins=60)
# plt.plot(hrs_trv['svfn'], hrs_trv['fdif'], 'o')
plt.savefig( os.path.join(output_dir, 'fdir_hist_hr.png'))
plt.show()

plt.figure()
plt.plot( hrs_trv['sian'], hrs_trv['fdir'], 'og')
plt.plot( np.ravel(rest_ip['SIAnorm']), np.ravel(fFMdir_field), 'o')
plt.plot( rmc_trv['sian'], rmc_trv['fdir'], 'or')
# plt.plot( np.ravel(fFMdir_field), rmc_trv['fdir'], 'og')
# plt.plot( np.ravel(fFMdir_field), np.ravel(fFMdir_field), '-k')
# plt.plot(hrs_trv['svfn'], hrs_trv['fdif'], 'o')
plt.savefig( os.path.join(output_dir, 'sian_vs_fdir_rmc.png'))
plt.show()




mycomp = 'fdir'
plt.figure()
# plt.hist(hrs_trv['sinssina'], bins=60, alpha = 0.9)
# plt.hist(rmc_trv['sinssina'], bins=60, alpha = 0.9)
plt.hist(rmc_res[mycomp], bins=100, alpha = 0.8, density=True, color='blue', label ='RMC')
# plt.hist(dbs_trv['fdir'], bins=60, alpha = 0.6, density=True, color='blue')
plt.hist(rmc_trv[mycomp], bins=100, alpha = 0.6, density=True, color='green', label = 'HRS-from RMC')
# plt.hist(hrs_trv[mycomp], bins=100, alpha = 0.5, density=True, color='red', label='HRS-from Prep')
plt.xlim(-2,4)
plt.ylim(0,1)
# plt.plot(hrs_trv['svfn'], hrs_trv['fdif'], 'o')
plt.legend()
plt.savefig( os.path.join(output_dir, '{}_hist_hr.png'.format(mycomp)))
plt.show()


print(ucla_trv[mycomp])
print( np.mean(rmc_res[mycomp]))
print( np.mean(rmc_trv[mycomp]) )
print( np.mean(hrs_trv[mycomp]) )
print( np.sum(frac* dbs_trv[mycomp]) )


for mycomp in ['fdir', 'frdir', 'fdif', 'frdif']:
    plt.figure()
    plt.plot(rmc_res[mycomp], rmc_trv[mycomp], 'o', markersize = 0.5)
    plt.plot(rmc_res[mycomp], rmc_res[mycomp], '-k')
    plt.savefig( os.path.join(output_dir, '{}_scatter_rmc.png'.format(mycomp)))
    plt.show()

# print( np.std(fFMdir_field))
# print( np.std(rmc_trv[mycomp]) )
# print( np.std(hrs_trv[mycomp]) )
# print( np.std(dbs_trv[mycomp]) ) # THIS HAS NO MEANING

# plt.figure()
# # plt.plot(hrs_trv['sian'], hrs_trv['fdir'], 'o')
# # plt.hist(hrs_trv['sian'], bins=60, alpha = 0.9)
# # plt.hist(hrs_trv['sinssina'], bins=60, alpha = 0.9)
# # plt.hist(rmc_trv['sinssina'], bins=60, alpha = 0.9)
# plt.plot(rmc_trv['fdir'], rmc_trv['fdir'], '-k')
# plt.plot(rmc_trv['fdir'], hrs_trv['fdir'], 'o')
# # plt.plot(hrs_trv['svfn'], hrs_trv['fdif'], 'o')
# plt.savefig( os.path.join(output_dir, 'fdir_scatter_hr.png'))
# plt.show()

