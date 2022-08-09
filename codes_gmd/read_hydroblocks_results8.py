
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


import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()

# scenarios = ['10', '25', '100'] # number of HRUs in each simulation
# scenarios = ['100'] # number of HRUs in each simulation
# scenarios = ['sia10', 'sia25', 'sia100'] # ONLY cos(sia) used in the clustering
# scenarios = ['notnorm10', 'notnorm25', 'notnorm100'] # ONLY cos(sia) used in the clustering


# datadir_all = os.path.join('..', '..', '..', 'Documents', 'res_hmc_light_p4')
datadir_all = os.path.join('..', '..', '..', 'Documents', 'res_hmc_light_k4')

DOMAINS = ['EastAlps']
# DOMAINS = ['EastAlps', 'Peru', 'Nepal']
ndomains = len(DOMAINS)

NHILLS = np.array([5, 10, 20, 50, 100])
# NHILLS = np.array([2, 5])
nnhills = len(NHILLS)
# nhills = [50]

# TERRVARS = []
# nterrvars = len(TERRVARS)


res_all = []

cosz = 0.7
adir = 0.5
phi = np.pi / 2
# flux_term = 'fdir'
do_averaging = False
aveblock = 55

all_res = []



# outfigdir = os.path.join(datadir, 'outfigdir')
# outdir = os.path.join(datadir, 'output')
outfigdir = os.path.join('//', 'home', 'enrico', 'Documents',
                       'dem_datasets', 'outfigdir')
read_data = True

if read_data:
    for ido, mydomain in enumerate(DOMAINS):
        for inh, mynhills in enumerate(NHILLS):


            datadir = os.path.join(datadir_all, 'res_{}_{}'.format(mydomain, mynhills))
            # datadir = os.path.join('..', '..', '..', 'Documents',
            #                        'res_{}'.format(mydomain),
            #                        'res_{}_light'.format(mydomain))

            modeldir = os.path.join('//', 'home', 'enrico', 'Documents',
                                       'dem_datasets', 'trained_models',
                                        'domain_EastAlps_buffer_0.1',
                                       'models_ave_{}'.format(aveblock),
                                       )

            res = til.read_tile_properties(datadir=datadir,
                                 do_averaging=do_averaging,
                                 aveblock= aveblock,
                                 cosz = cosz, phi = phi, adir=adir,
                                 modeldir=modeldir)
            all_res.append(res)



    with open(os.path.join(outfigdir, 'flux_stats_k4.pkl'), 'wb') as f:
        pickle.dump(all_res, f)

else:


    with open(os.path.join(outfigdir, 'flux_stats_k4.pkl'), 'rb') as f:
        all_res = pickle.load(f)

# load coords
X, Y = np.meshgrid(all_res[0]['xlon_ea'], all_res[0]['ylat_ea']) # coords equal area in latlon
Xl, Yl = np.meshgrid(all_res[0]['xlon_latlon'], all_res[0]['ylat_latlon']) # coords equal area in latlon



    # cm0 = axes[1].pcolormesh(X, Y, res['mappedtile_fdir'], cmap='jet', vmin=-1, vmax=1)
#         axes.set_title('{}'.format(mydomain))
#         # axes[1].set_title(r'tile-by-tile predictions ($n_T = {}$)'.format(res['ntiles']))
#         # axes.set_xlabel('x [km]')
#         # axes.set_ylabel('y [km]')
#         axes.set_xlabel('Longitude')
#         axes.set_ylabel('Latitude')
#         cbar = fig.colorbar(cm1)
#         cbar.set_label(r'Elevation [m]')
#         plt.savefig( os.path.join(outfigdir, 'elev_map_{}.png'.format(mydomain)))
#         plt.show()
#

matplotlib.use('Qt5Agg') # dyn show plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
hrus = all_res[0]['mappedtile_hrus'].copy()
hills = all_res[0]['mappedtile_hills'].copy().astype(np.int)
hrus[hrus<0] = -1
hills[hills<0] = -1
# hills[hills>10000] = -1
cm0 = axes[0].pcolormesh(X, Y, hrus , cmap='Pastel1')
cm1 = axes[1].imshow(hills , cmap='Pastel1')
# cm0 = axes[1].pcolormesh(X, Y, res['mappedtile_fdir'], cmap='jet', vmin=-1, vmax=1)
# axes[0].set_title('{}'.format(mydomain))
# axes.set_title('{}'.format(mydomain))
# axes[1].set_title(r'tile-by-tile predictions ($n_T = {}$)'.format(res['ntiles']))
# axes.set_xlabel('x [km]')
# axes.set_ylabel('y [km]')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
cbar0 = fig.colorbar(cm0, ax=axes[0])
cbar1 = fig.colorbar(cm1, ax=axes[1])
cbar0.set_label(r'Tile')
cbar1.set_label(r'Chill')
# plt.savefig( os.path.join(outfigdir, 'varplot_map_{}_{}.png'.format(mydomain, myvar)))
plt.show()



matplotlib.use('Qt5Agg') # dyn show plots
plt.figure()
plt.plot( np.ravel(all_res[0]['map_frdifn'][:100, :100]), np.ravel(all_res[0]['map_frdirn'][:100, :100]), 'o')
plt.plot( np.ravel(all_res[0]['map_frdirn'][:100, :100]), np.ravel(all_res[0]['map_frdirn'][:100, :100]), 'k')
plt.show()

# limy = 800 # use -1 for entire plot
# limx = 600 # use -1 for entire plot
limy = -1 # use -1 for entire plot
limx = -1 # use -1 for entire plo

bnd = 1
matplotlib.use('Agg') # dyn show plots


allvars = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn', 'sia', 'tcf', 'svf']
# BND =     [1,        1,       1,         1,        1,      3,     1,      1 ]

for iv, myvar in enumerate(allvars):
    # bnd = BND[iv]
    up_bnd = np.quantile(all_res[0]['map_{}'.format(myvar)][:limy, :limx], 0.95)
    do_bnd = np.quantile(all_res[0]['map_{}'.format(myvar)][:limy, :limx], 0.05)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    cm0 = axes[0,0].pcolormesh(X[:limy, :limx], Y[:limy, :limx], all_res[0]['map_{}'.format(myvar)][:limy, :limx], cmap='jet',        vmin = do_bnd, vmax = up_bnd)
    cm1 = axes[0,1].pcolormesh(X[:limy, :limx], Y[:limy, :limx], all_res[0]['mappedtile_{}'.format(myvar)][:limy, :limx], cmap='jet', vmin = do_bnd, vmax = up_bnd)
    cm2 = axes[1,0].pcolormesh(X[:limy, :limx], Y[:limy, :limx], all_res[1]['mappedtile_{}'.format(myvar)][:limy, :limx], cmap='jet', vmin = do_bnd, vmax = up_bnd)
    cm3 = axes[1,1].pcolormesh(X[:limy, :limx], Y[:limy, :limx], all_res[2]['mappedtile_{}'.format(myvar)][:limy, :limx], cmap='jet', vmin = do_bnd, vmax = up_bnd)

    axes[0,0].set_title('High res. field')
    axes[0,1].set_title('$k = 5$ tiles')
    axes[1,0].set_title('$k = 20$ tiles')
    axes[1,1].set_title('$k = 50$ tiles')

    axes[0,0].set_ylabel('Latitude')
    axes[0,0].set_xlabel('Longitude')
    axes[1,0].set_ylabel('Latitude')
    axes[1,0].set_xlabel('Longitude')

    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].set_xlabel('Longitude')

    cbar0 = fig.colorbar(cm0, ax=axes[0,0], extend='both')
    cbar1 = fig.colorbar(cm1, ax=axes[0,1], extend='both')
    cbar2 = fig.colorbar(cm2, ax=axes[1,0], extend='both')
    cbar3 = fig.colorbar(cm3, ax=axes[1,1], extend='both')

    plt.tight_layout()
    plt.savefig( os.path.join(outfigdir, 'varplot_map_{}.png'.format(myvar)))

    plt.show()
# endplot


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,8))

A = np.ravel(all_res[2]['map_sia'])
B = np.ravel(all_res[2]['mappedtile_sia'])
# C = np.logical_not(np.isnan(B))
# A = A[C]
# B = B[C]


# len(B[~np.isnan(B)])

# axes[0].plot(A, B)
# axes[0].plot( np.ravel(all_res[0]['map_fdir']), np.ravel(all_res[0]['mappedtile_fdir']) )
# axes[1].plot( np.ravel(all_res[0]['map_fdir']), np.ravel(all_res[1]['mappedtile_fdir']))
# axes[2].plot( np.ravel(all_res[0]['map_fdir']), np.ravel(all_res[2]['mappedtile_fdir']))

# axes[0,0].set_title('High res. field')
# axes[0,1].set_title('$k = 5$ tiles')
# axes[1,0].set_title('$k = 20$ tiles')
# axes[1,1].set_title('$k = 50$ tiles')

# cbar0 = fig.colorbar(cm0, ax=axes[0,0], extend='both')

# gridsize1 = 18
# plt.figure()
# plt.hexbin(A, B,
#         gridsize = gridsize1, cmap = plt.cm.Greens, bins = 'log')
# # plt.plot([np.min(A), np.max(A)], [np.min(A), np.max(A)], '--k')
# plt.show()
#

# plt.show()


gridsize1 = 14
vars = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
nvars = len(vars)

# NHILLS = np.array([5, 20, 50])
# nnhills = len(NHILLS)
#



# nrowsp = 5
# ncolsp = 3
fig, axes = plt.subplots(nrows=nvars, ncols=nnhills, figsize=(18, 22))

countp = 0
for j, myvar in enumerate(vars):
    for i, mynhill in enumerate(NHILLS):

        axes[j,i].hexbin( np.ravel(all_res[i]['map_{}'.format(myvar)]),
                          np.ravel(all_res[i]['mappedtile_{}'.format(myvar)]),
                         gridsize = gridsize1, cmap = plt.cm.Greens, bins='log')
        # axes[j,i].scatter( np.nanmean(res_all[i]['hrt_pred_{}'.format(FLUX_TERMS[j])]),
        #                    np.nanmean(res_all[i]['hru_pred_{}'.format(FLUX_TERMS[j])]),
        #                    marker='d', c='r', s=35)
        minx = np.nanmin( np.ravel(all_res[i]['map_{}'.format(myvar)]))
        maxx = np.nanmax( np.ravel(all_res[i]['map_{}'.format(myvar)]))
        miny = np.nanmin( np.ravel(all_res[i]['mappedtile_{}'.format(myvar)]))
        maxy = np.nanmax( np.ravel(all_res[i]['mappedtile_{}'.format(myvar)]))
        minlx = min(minx, miny)
        maxlx = min(maxx, maxy)
        # axes[2,0].plot([minlx, maxlx], [minlx, maxlx], 'k')
        # axes[j,i].plot([miny, maxy], [miny, maxy], 'k')
        axes[j,i].plot([minx, maxx], [minx, maxx], 'k')


        # axes[2,0].plot([-0.5, 0.5], [-0.5, 0.5], 'k')
        if i==0:
            pad = 7
            axes[j, i].annotate('{}'.format(myvar),
                                xy=(0, 0.5), xytext=(-axes[j, i].yaxis.labelpad - pad, 0),
                                xycoords=axes[j,i].yaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center')

        if j != len(vars) - 1:
            axes[j, i].xaxis.set_ticklabels(
                [])  # x-ticklabels only on bottom row
        else:
            axes[j, i].set_xlabel('high-res [-]')

        if i == 0:
            axes[j, i].set_ylabel('cluster [-]')
        # if j == 4:
        if j == 0:
            axes[j, i].set_title('k = {} hillslopes'.format(mynhill))

        axes[j,i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                       transform=axes[j,i].transAxes,
                       size=20, weight='bold')

        countp += 1

plt.tight_layout()
plt.savefig(os.path.join(outfigdir, 'res_scatter_hrus_new.png'), dpi = 300)
# endplot


#
#         print(hrus.shape)
#         tilenumb = np.unique(res['mappedtile_hrus'])
#         ntiles = len(tilenumb)
#         hillnumb = len(np.unique(hills))
#         print(ntiles, hillnumb)
#         # ntiles = len(tilenumb
#
#         # plt.figure()
#         # # plt.imshow(hills, cmap='Pastel1')
#         # plt.hist(np.ravel(hills), bins=100)
#         # plt.colorbar()
#         # plt.show()
#         # #
#         # np.size(np.unique(hills))
#
#         # sia = res['mappedtile_sia']
#         # sia = res['map_sia']
#         sia = res['map_tcf']
#         print(sia.shape)
#
#         plt.figure()
#         plt.hist(np.ravel(res['map_sinscosa']), bins=100)
#         plt.show()
#
#         sia_1 = sia[hrus==1]
#         sia_2 = sia[hrus==2]
#         sia_10 = sia[hrus==10]
#         sia_39 = sia[hrus==39]
#         sia_700 = sia[hrus==700]
#         sia_800 = sia[hrus==800]
#
#         plt.figure()
#         plt.hist(sia_1,  bins=60,  alpha=0.9, density=True)
#         plt.hist(sia_2,  bins=60,  alpha=0.9, density=True)
#         plt.hist(sia_10, bins=60, alpha = 0.4, density=True)
#         plt.hist(sia_39, bins=60, alpha = 0.4, density=True)
#         plt.hist(sia_700, bins=60, alpha = 0.4, density=True)
#         plt.hist(sia_800, bins=60, alpha = 0.4, density=True)
#         plt.show()
#
#
#
#
# matplotlib.use('Qt5Agg') # dyn show plots
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17,8))
# cm1 = axes[0].pcolormesh(X, Y, res['map_fdir'], cmap='jet', vmin=-1, vmax=1)
# cm0 = axes[1].pcolormesh(X, Y, res['mappedtile_fdir'], cmap='jet', vmin=-1, vmax=1)
# axes[0].set_title('High-res predictions')
# axes[1].set_title(r'tile-by-tile predictions ($n_T = {}$)'.format(res['ntiles']))
# axes[0].set_xlabel('x [km]')
# axes[0].set_ylabel('y [km]')
#
# axes[1].set_xlabel('x [km]')
# axes[1].set_ylabel('y [km]')
# cbar = fig.colorbar(cm0)
# cbar.set_label(r'(3D - PP)/PP')
# plt.savefig( os.path.join(outfigdir, 'comp_fdir.png'))
# plt.show()
#
#
# matplotlib.use('Qt5Agg') # dyn show plots
# bnd = 0.5
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17,8))
# cm1 = axes[0].pcolormesh(X, Y, res['map_fdif'], cmap='jet', vmin=-bnd, vmax=bnd)
# cm0 = axes[1].pcolormesh(X, Y, res['mappedtile_fdif'], cmap='jet', vmin=-bnd, vmax=bnd)
# axes[0].set_title('High-res predictions')
# axes[1].set_title(r'tile-by-tile predictions ($n_T = {}$)'.format(res['ntiles']))
# axes[0].set_xlabel('x [km]')
# axes[0].set_ylabel('y [km]')
#
# axes[1].set_xlabel('x [km]')
# axes[1].set_ylabel('y [km]')
# cbar = fig.colorbar(cm0)
# cbar.set_label(r'(3D - PP)/PP')
# plt.savefig( os.path.join(outfigdir, 'comp_fdif.png'))
# plt.show()
#
#
# matplotlib.use('Qt5Agg') # dyn show plots
# bnd = 0.5
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17,8))
# cm1 = axes[0].pcolormesh(X, Y, res['map_fcoupn'], cmap='jet', vmin=-bnd, vmax=bnd)
# cm0 = axes[1].pcolormesh(X, Y, res['mappedtile_fcoupn'], cmap='jet', vmin=-bnd, vmax=bnd)
# axes[0].set_title('High-res predictions')
# axes[1].set_title(r'tile-by-tile predictions ($n_T = {}$)'.format(res['ntiles']))
# axes[0].set_xlabel('x [km]')
# axes[0].set_ylabel('y [km]')
#
# axes[1].set_xlabel('x [km]')
# axes[1].set_ylabel('y [km]')
# cbar = fig.colorbar(cm0)
# cbar.set_label(r'(3D - PP)/PP')
# plt.savefig( os.path.join(outfigdir, 'comp_fcoupn.png'))
# plt.show()
#
# matplotlib.use('Qt5Agg') # dyn show plots
# plt.figure()
# plt.imshow(res['mappedtile_tcf'])
# plt.imshow(res['map_tcf'])
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.hist( np.ravel(  res['map_fdir'])  , bins=50 )
# plt.show()
#
# plt.figure()
# plt.plot(  np.ravel(res['map_fdir']), np.ravel(res['mappedtile_fdir']), 'o')
# # plt.plot(  np.ravel(res['map_fdif']), np.ravel(res['mappedtile_fdif']), 'o')
# # plt.plot(  np.ravel(res['map_frdirn']), np.ravel(res['mappedtile_frdirn']), 'o')
# # plt.plot(  np.ravel(res['map_fcoupn']), np.ravel(res['mappedtile_fcoupn']), 'o')
# plt.plot( [-1, 1], [-1, 1], 'k')
# plt.show()
#
# plt.figure()
# plt.plot(  np.ravel(res['map_sia']), np.ravel(res['mappedtile_sia']), 'o')
# # plt.plot(  np.ravel(res['map_svf']), np.ravel(res['mappedtile_svf']), 'o')
# # plt.plot(  np.ravel(res['map_tcf']), np.ravel(res['mappedtile_tcf']), 'o')
# # plt.plot(  np.ravel(res['map_ele']), np.ravel(res['mappedtile_ele']), 'o')
# # plt.plot(  np.ravel(res['map_sde']), np.ravel(res['mappedtile_sde']), 'o')
# plt.plot( [0, 1], [0, 1], 'k')
# plt.show()
