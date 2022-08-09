
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

# scenarios = ['Peru']
scenarios = ['EastAlps', 'Nepal', 'Peru']
nhills = 2

res_all = []

cosz = 0.7
adir = 0.5
phi = np.pi / 2
# flux_term = 'fdir'
do_averaging = False
aveblock = 55

# outfigdir = os.path.join(datadir, 'outfigdir')
# outdir = os.path.join(datadir, 'output')
outfigdir = os.path.join('//', 'home', 'enrico', 'Documents',
                       'dem_datasets', 'outfigdir')
res_all = {}
for isc, sc in enumerate(scenarios):

    datadir_all = os.path.join('..', '..', '..', 'Documents',
                               'res_hmc_light_p4')
    datadir = os.path.join(datadir_all, 'res_{}_{}'.format(sc, nhills))

    # datadir = os.path.join('..', '..', '..', 'Documents', 'res_{}'.format(sc))
    # datadir = os.path.join('..', '..', '..', 'Documents', 'res_all_hmc', 'res_Peru_10')
    #
    # res = til.read_tile_properties(datadir=datadir, do_averaging=do_averaging,
    #                      aveblock= aveblock,
    #                      cosz = cosz, phi = phi, adir=adir,
    #                      modeldir=None)

    modeldir = os.path.join('//', 'home', 'enrico', 'Documents',
                            'dem_datasets', 'trained_models',
                            'domain_EastAlps_buffer_0.1',
                            'models_ave_{}'.format(aveblock),
                            )

    res = til.read_tile_properties(datadir=datadir,
                                   do_averaging=do_averaging,
                                   aveblock=aveblock,
                                   cosz=cosz, phi=phi, adir=adir,
                                   modeldir=modeldir)
    # # save results on
    # res_all['{}_{}'.format(sc, nhills)] = res


    # res['mappedtile_sia'].shape
    # res['map_sia'].shape

    # show results for a subset::

    # equal area map - distance in [km]

    # load coords
    X, Y = np.meshgrid(res['xlon_ea'], res['ylat_ea'])  # coords equal area in latlon
    Xl, Yl = np.meshgrid(res['xlon_latlon'], res['ylat_latlon'])  # coords equal area in latlon

    # X = np.flipud( np.abs(X)  )
    # Y = np.flipud( np.abs(Y)  )

    matplotlib.use('Qt5Agg') # dyn show plots
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    # cm1 = axes[0].pcolormesh(X, Y, res['map_fdir'], cmap='jet', vmin=-1, vmax=1)
    cm0 = axes.pcolormesh(X, Y, res['map_ele'], cmap='terrain')
    axes.set_title('{}'.format(sc))

    if Y[0, 0]<0:
        axes.set_ylabel('Latitude [S]')
    else:
        axes.set_ylabel('Latitude [N]')

    if X[0, 0] < 0:
        axes.set_xlabel('Longitude [W]')
    else:
        axes.set_xlabel('Longitude [E]')

    # ylabs = abs(axes.get_yticks())
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.ticker as ticker

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        mylab = -x if x < 0 else x
        label = '%.1f' % mylab
        return label

    # axes.set_yticklabels(ylabs)
    axes.yaxis.set_major_formatter(major_formatter)
    axes.xaxis.set_major_formatter(major_formatter)
    # axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    cbar = fig.colorbar(cm0)
    cbar.set_label(r'Elevation [m m.s.l.]')
    # plt.tight_layout()
    plt.savefig( os.path.join(outfigdir, 'elevmap_{}.png'.format(sc)), dpi = 300)
    plt.show()



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
# # plt.plot(  np.ravel(res['map_sia']), np.ravel(res['mappedtile_sia']), 'o')
# plt.plot(  np.ravel(res['map_svf']), np.ravel(res['mappedtile_svf']), 'o')
# # plt.plot(  np.ravel(res['map_tcf']), np.ravel(res['mappedtile_tcf']), 'o')
# # plt.plot(  np.ravel(res['map_ele']), np.ravel(res['mappedtile_ele']), 'o')
# # plt.plot(  np.ravel(res['map_sde']), np.ravel(res['mappedtile_sde']), 'o')
# plt.plot( [0, 1], [0, 1], 'k')
# plt.show()
#
# res['map_fdif'].shape
# res['mappedtile_fdif'].shape