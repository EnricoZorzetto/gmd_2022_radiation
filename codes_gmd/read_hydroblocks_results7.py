
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


import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('MacOSX') # dyn show plots
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()

# scenarios = ['10', '25', '100'] # number of HRUs in each simulation
# scenarios = ['100'] # number of HRUs in each simulation
# scenarios = ['sia10', 'sia25', 'sia100'] # ONLY cos(sia) used in the clustering
# scenarios = ['notnorm10', 'notnorm25', 'notnorm100'] # ONLY cos(sia) used in the clustering

# tile_type = 'p4'
tile_type = 'k4'

# datadir_all = os.path.join('..', '..', '..', 'Documents', 'res_hmc_light_p4')
# datadir_all = os.path.join('..', '..', '..', 'Documents', 'res_hmc_light_p2')
# datadir_all = os.path.join('..', '..', '..', 'Documents', 'res_hmc_light_{}'.format(tile_type))
datadir_all = os.path.join('/Users', 'ez6263','Documents', 'res_hmc_light_{}'.format(tile_type))
os.listdir(datadir_all)

# DOMAINS = ['Peru', 'Nepal']
DOMAINS = ['EastAlps', 'Peru', 'Nepal']
ndomains = len(DOMAINS)

NHILLS = np.array([2, 5, 10, 20, 50, 100, 200])
# NHILLS = np.array([2, 5])
nnhills = len(NHILLS)
# nhills = [50]

# TERRVARS = []
# nterrvars = len(TERRVARS)


res_all = []

cosz = 0.7
adir = 0.3
phi = np.pi / 2
# flux_term = 'fdir'
do_averaging = False
aveblock = 55

# outfigdir = os.path.join(datadir, 'outfigdir')
# outdir = os.path.join(datadir, 'output')
# outfigdir = os.path.join('//', 'home', 'enrico', 'Documents',
#                        'dem_datasets', 'outfigdir')

outfigdir = os.path.join('/Users', 'ez6263','Documents', 'gmd_2021_grids_output', 'res_old_codes')
os.system("mkdir -p {}".format(outfigdir))
os.listdir(outfigdir)

# STDV_SIA = np.zeros(nnhills)
# STDV_FDIR = np.zeros(nnhills)
#
# STDV_SIA_HR = np.zeros(nnhills)
# STDV_FDIR_HR = np.zeros(nnhills)


VARS = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn', 'svf', 'tcf', 'sia']
nvars = len(VARS)


# save results::

STATS = ['mean', 'stdv', 'skew', 'kurt']
nstats = len(STATS)

resmat = np.zeros((nnhills, ndomains, nvars, nstats, 2))
# TERR_STATS = np.zeros((nnhills, ndomains, nterrvars, nstats))

read_data = True

if read_data:
    for ido, mydomain in enumerate(DOMAINS):
        for inh, mynhills in enumerate(NHILLS):


            datadir = os.path.join(datadir_all, 'res_{}_{}'.format(mydomain, mynhills))
            # datadir = os.path.join('..', '..', '..', 'Documents',
            #                        'res_{}'.format(mydomain),
            #                        'res_{}_light'.format(mydomain))

            # modeldir = os.path.join('//', 'home', 'enrico', 'Documents',
            #                            'dem_datasets', 'trained_models',
            #                             'domain_EastAlps_buffer_0.1',
            #                            'models_ave_{}'.format(aveblock),
            #                            )

            modeldir = os.path.join('/Users', 'ez6263', 'Documents',
                                    'dem_datasets', 'res_Peru_vs_EastAlps',
                                    'trained_models',
                                    'domain_EastAlps_buffer_0.35',
                                    'models_ave_{}'.format(aveblock),
                                    )

            res = til.read_tile_properties(datadir=datadir,
                                 do_averaging=do_averaging,
                                 aveblock= aveblock,
                                 cosz = cosz, phi = phi, adir=adir,
                                 modeldir=modeldir)

            # X, Y = np.meshgrid(res['x_ea'], res['y_ea']) # coords equal area in [km]
            X, Y = np.meshgrid(res['xlon_ea'], res['ylat_ea']) # coords equal area in latlon
            Xl, Yl = np.meshgrid(res['xlon_latlon'], res['ylat_latlon']) # coords equal area in latlon
            # Y, X = np.meshgrid(res['ylat_ea'], res['xlon_ea']) # coords equal area in latlon

            # STDV_SIA[int] = np.nanstd(res['mappedtile_sia'])
            # STDV_FDIR[int] = np.nanstd(res['mappedtile_fdir'])
            # STDV_SIA_HR[int] = np.nanstd(res['map_sia'])
            # STDV_FDIR_HR[int] = np.nanstd(res['map_fdir'])

            for ivar, myvar in enumerate(VARS):
                                 # stat 0-3 # tile or hr 0-1
                resmat[inh, ido, ivar, 0, 0] = np.nanmean( res['mappedtile_{}'.format(myvar)])
                resmat[inh, ido, ivar, 0, 1] = np.nanmean( res['map_{}'.format(myvar)])
                resmat[inh, ido, ivar, 1, 0] = np.nanstd( res['mappedtile_{}'.format(myvar)])
                resmat[inh, ido, ivar, 1, 1] = np.nanstd( res['map_{}'.format(myvar)])

                resmat[inh, ido, ivar, 2, 0] = til.skew( res['mappedtile_{}'.format(myvar)])
                resmat[inh, ido, ivar, 2, 1] = til.skew( res['map_{}'.format(myvar)])
                resmat[inh, ido, ivar, 3, 0] = til.kurt( res['mappedtile_{}'.format(myvar)])
                resmat[inh, ido, ivar, 3, 1] = til.kurt( res['map_{}'.format(myvar)])

                resp =  til.kurt( res['map_{}'.format(myvar)])

    # create array with results

    da = xr.DataArray(
        data=resmat,
        dims=["nhills", "domains", "vars", "stats", "tile_highres"],
        coords=dict(
            nhills=(["nhills"], NHILLS),
            domains=(["domains"], DOMAINS),
            vars=(["vars"], VARS),
            stats=(["stats"], STATS),
            tile_highres=(["tile_highres"], ["tile", "highres"])
            # time=time,
            # reference_time=reference_time,
        ),
        attrs=dict(
            description="Statistics of fluxes and terrain properties.",
            units="dimenstionless values",
        ),
    )

    with open( os.path.join(outfigdir, 'flux_stats_{}.pkl'.format(tile_type)), 'wb') as f:
        pickle.dump(da, f)

else:

    with open(os.path.join(outfigdir, 'flux_stats_{}.pkl'.format(tile_type)), 'rb') as f:
        da = pickle.load(f)

    with open(os.path.join(outfigdir, 'flux_stats_k4.pkl'),'rb') as f:
        dak = pickle.load(f)

    with open(os.path.join(outfigdir, 'flux_stats_p4.pkl'),'rb') as f:
        dap = pickle.load(f)

# matplotlib.use('Qt5Agg') # dyn show plots
# matplotlib.use('MacOSX') # dyn show plots
mystat = 'skew'
plt.figure()
plt.plot(da.coords['nhills'], da.loc[:, 'Nepal', 'fdir', mystat, 'tile'],'o')
y_highres =  da.loc[:, 'Nepal', 'fdir', mystat, 'highres']
plt.plot(da.coords['nhills'], y_highres)
plt.ylim([0, np.max(y_highres)*1.1])
plt.show()



# matplotlib.use('Qt5Agg') # dyn show plots
nvars =  len(da.coords['vars'])
stats0 = da.coords['stats'].values
mystats = ['stdv', 'skew', 'kurt']
stats = [ stat for stat in list(stats0) if stat in mystats]
stats_symb = [r'st. dev. $ \sigma_x$', r'skewness $\gamma_x$', r'kurtosis $\xi_x$']
nstats = len(stats)
ndoms =  len(da.coords['domains'])
mystat = 'skew'
vars = da.coords['vars'].values
fig, axes = plt.subplots(nrows=nvars, ncols=nstats, figsize = (22, 16))
# plt.grid(True)
for i, myvar in enumerate( vars ):
    for j, mystat in enumerate(  stats):
        # axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Nepal', myvar, mystat, 'tile'],'-o')
        y_highres_Nepal = da.loc[:, 'Nepal', myvar, mystat, 'highres']
        y_highres_Peru = da.loc[:, 'Peru', myvar, mystat, 'highres']
        y_highres_EastAlps = da.loc[:, 'EastAlps', myvar, mystat, 'highres']
        axes[i, j].plot(da.coords['nhills'], np.ones(len(da.coords['nhills'])), '--k')
        axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Nepal', myvar, mystat, 'tile'].values/y_highres_Nepal,    '-o', color='blue',  label='Nepal')
        axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Peru', myvar, mystat, 'tile'].values/y_highres_Peru,     '-*', color='red',   label = 'Peru')
        axes[i, j].plot(da.coords['nhills'], da.loc[:, 'EastAlps', myvar, mystat, 'tile'].values/y_highres_EastAlps, '-^', color='green', label = 'EastAlps')
        if mystat != 'mean' and mystat != 'skew':
            axes[i, j].set_ylim(bottom=0.0)
        # else:
        #     axes[i, j].set_ylim([ min( )*1.1, max()*1.1 ])
        # axes[i, j].set_title('{}, {}'.format(myvar, mystat))
        axes[i, j].set_xscale('log')
        axes[i,j].grid(True)
        if i == 0 and j ==  0:
            # axes[i, j].legend(loc='lower right', ncol=3) # legend only on first plot
            axes[i, j].legend(loc='lower right', ncol=5, bbox_to_anchor=(3.0, 1.45))  # legend only on first plot
        if i == 0:
            axes[i, j].set_title(stats_symb[j]) # titles only on top panels

        if i != len(vars)-1:
            axes[i, j].xaxis.set_ticklabels([]) # x-ticklabels only on bottom row
        else:
            axes[i, j].set_xlabel(r'$k$')
        if j == 0:
            # axes[i, j].set_ylabel(r'$ \frac{{{}_{{k}}}}{{{{}_{HR}}}}}$'.format(myvar), rotation = 0)
            axes[i, j].set_ylabel(r'$   {}_k/{}_{{HR}}$'.format(myvar, myvar), rotation=0, labelpad = 30)
            axes[i, j].set_ylabel(r'$   \frac{{   {}_k }}{{  {}_{{HR}}  }}$'.format(myvar, myvar), rotation=0)
plt.savefig(os.path.join(outfigdir, 'conv_tiles_p2.png'), dpi = 300)
plt.show()









# matplotlib.use('Qt5Agg') # dyn show plots
vars = da.coords['vars'].values
vars = vars[:5]
print(vars)
nvars =  len(vars)
stats0 = da.coords['stats'].values
mystats = ['stdv', 'skew', 'kurt']
stats = [ stat for stat in list(stats0) if stat in mystats]
stats_symb = [r'st. dev. $ \sigma_x$', r'skewness $\gamma_x$', r'kurtosis $\xi_x$']
nstats = len(stats)
ndoms =  len(da.coords['domains'])
mystat = 'skew'
fig, axes = plt.subplots(nrows=nvars, ncols=nstats, figsize = (16, 20))
# plt.grid(True)
for i, myvar in enumerate( vars ):
    for j, mystat in enumerate(  stats):
        # axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Nepal', myvar, mystat, 'tile'],'-o')
        y_highres_Nepal_p = dap.loc[:, 'Nepal', myvar, mystat, 'highres']
        y_highres_Peru_p = dap.loc[:, 'Peru', myvar, mystat, 'highres']
        y_highres_EastAlps_p = dap.loc[:, 'EastAlps', myvar, mystat, 'highres']

        y_highres_Nepal_k = dak.loc[:, 'Nepal', myvar, mystat, 'highres']
        y_highres_Peru_k = dak.loc[:, 'Peru', myvar, mystat, 'highres']
        y_highres_EastAlps_k = dak.loc[:, 'EastAlps', myvar, mystat, 'highres']

        axes[i, j].plot(da.coords['nhills'], np.ones(len(da.coords['nhills'])), '--k')

        axes[i, j].plot(dap.coords['nhills'], dap.loc[:, 'Nepal', myvar, mystat, 'tile'].values/y_highres_Nepal_p,    '-o', color='blue',  label='varying $k$')
        axes[i, j].plot(dap.coords['nhills'], dap.loc[:, 'Peru', myvar, mystat, 'tile'].values/y_highres_Peru_p,     '-*', color='red')
        axes[i, j].plot(dap.coords['nhills'], dap.loc[:, 'EastAlps', myvar, mystat, 'tile'].values/y_highres_EastAlps_p, '-^', color='green')

        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'Nepal', myvar, mystat,'tile'].values / y_highres_Nepal_k, '--o', color='blue', label='varying $p$')


        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'Nepal', myvar, mystat,'tile'].values / y_highres_Nepal_k, '--o', color='blue', label='Nepal')
        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'Peru', myvar, mystat, 'tile'].values / y_highres_Peru_k,'--*', color='red', label = 'Peru')
        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'EastAlps', myvar, mystat, 'tile'].values / y_highres_EastAlps_k, '--^',color='green', label = 'EastAlps')

        if mystat != 'mean' and mystat != 'skew':
            axes[i, j].set_ylim(bottom=0.0)
        # else:
        #     axes[i, j].set_ylim([ min( )*1.1, max()*1.1 ])
        # axes[i, j].set_title('{}, {}'.format(myvar, mystat))
        axes[i, j].set_xscale('log')
        axes[i,j].grid(True)
        if i == 0 and j ==  0:
            # axes[i, j].legend(loc='lower right', ncol=2) # legend only on first plot
            axes[i, j].legend(loc='lower right', ncol=5, bbox_to_anchor=(3.0, 1.45))  # legend only on first plot
        if i == 0:
            axes[i, j].set_title(stats_symb[j]) # titles only on top panels

        if i != len(vars)-1:
            axes[i, j].xaxis.set_ticklabels([]) # x-ticklabels only on bottom row
        else:
            axes[i, j].set_xlabel(r'$n_{tiles}/4$')
        if j == 0:
            # axes[i, j].set_ylabel(r'$ \frac{{{}_{{k}}}}{{{{}_{HR}}}}}$'.format(myvar), rotation = 0)
            axes[i, j].set_ylabel(r'$   {}_k/{}_{{HR}}$'.format(myvar, myvar), rotation=0, labelpad = 30)
            axes[i, j].set_ylabel(r'$   \frac{{   {}_k }}{{  {}_{{HR}}  }}$'.format(myvar, myvar), rotation=0)
plt.savefig(os.path.join(outfigdir, 'conv_tiles_p4k4.png'), dpi = 300)
plt.show()




# matplotlib.use('Qt5Agg') # dyn show plots
vars = da.coords['vars'].values
vars = vars[5:]
print(vars)
nvars =  len(vars)
stats0 = da.coords['stats'].values
mystats = ['stdv', 'skew', 'kurt']
stats = [ stat for stat in list(stats0) if stat in mystats]
stats_symb = [r'st. dev. $ \sigma_x$', r'skewness $\gamma_x$', r'kurtosis $\xi_x$']
nstats = len(stats)
ndoms =  len(da.coords['domains'])
mystat = 'skew'
fig, axes = plt.subplots(nrows=nvars, ncols=nstats, figsize = (16, 20))
# plt.grid(True)
for i, myvar in enumerate( vars ):
    for j, mystat in enumerate(  stats):
        # axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Nepal', myvar, mystat, 'tile'],'-o')
        y_highres_Nepal_p = dap.loc[:, 'Nepal', myvar, mystat, 'highres']
        y_highres_Peru_p = dap.loc[:, 'Peru', myvar, mystat, 'highres']
        y_highres_EastAlps_p = dap.loc[:, 'EastAlps', myvar, mystat, 'highres']

        y_highres_Nepal_k = dak.loc[:, 'Nepal', myvar, mystat, 'highres']
        y_highres_Peru_k = dak.loc[:, 'Peru', myvar, mystat, 'highres']
        y_highres_EastAlps_k = dak.loc[:, 'EastAlps', myvar, mystat, 'highres']

        axes[i, j].plot(da.coords['nhills'], np.ones(len(da.coords['nhills'])), '--k')

        axes[i, j].plot(dap.coords['nhills'], dap.loc[:, 'Nepal', myvar, mystat, 'tile'].values/y_highres_Nepal_p,    '-o', color='blue',  label='varying $k$')
        axes[i, j].plot(dap.coords['nhills'], dap.loc[:, 'Peru', myvar, mystat, 'tile'].values/y_highres_Peru_p,     '-*', color='red')
        axes[i, j].plot(dap.coords['nhills'], dap.loc[:, 'EastAlps', myvar, mystat, 'tile'].values/y_highres_EastAlps_p, '-^', color='green')

        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'Nepal', myvar, mystat,'tile'].values / y_highres_Nepal_k, '--o', color='blue', label='varying $p$')

        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'Nepal', myvar, mystat,'tile'].values / y_highres_Nepal_k, '--o', color='blue', label='Nepal')
        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'Peru', myvar, mystat, 'tile'].values / y_highres_Peru_k,'--*', color='red', label = 'Peru')
        axes[i, j].plot(dak.coords['nhills'], dak.loc[:, 'EastAlps', myvar, mystat, 'tile'].values / y_highres_EastAlps_k, '--^',color='green', label = 'EastAlps')

        if mystat != 'mean' and mystat != 'skew':
            axes[i, j].set_ylim(bottom=0.0)
        # else:
        #     axes[i, j].set_ylim([ min( )*1.1, max()*1.1 ])
        # axes[i, j].set_title('{}, {}'.format(myvar, mystat))
        axes[i, j].set_xscale('log')
        axes[i,j].grid(True)
        if i == 0 and j ==  0:
            # axes[i, j].legend(loc='lower right', ncol=3) # legend only on first plot
            axes[i, j].legend(loc='lower right', ncol=5, bbox_to_anchor=(3.0, 1.25))  # legend only on first plot
        if i == 0:
            axes[i, j].set_title(stats_symb[j]) # titles only on top panels

        if i != len(vars)-1:
            axes[i, j].xaxis.set_ticklabels([]) # x-ticklabels only on bottom row
        else:
            axes[i, j].set_xlabel(r'$n_{tiles}/4$')
        if j == 0:
            # axes[i, j].set_ylabel(r'$ \frac{{{}_{{k}}}}{{{{}_{HR}}}}}$'.format(myvar), rotation = 0)
            axes[i, j].set_ylabel(r'$   {}_k/{}_{{HR}}$'.format(myvar, myvar), rotation=0, labelpad = 30)
            axes[i, j].set_ylabel(r'$   \frac{{   {}_k }}{{  {}_{{HR}}  }}$'.format(myvar, myvar), rotation=0)
plt.savefig(os.path.join(outfigdir, 'conv_tiles_p4k4_terrain.png'), dpi = 300)
plt.show()

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
#         matplotlib.use('Qt5Agg') # dyn show plots
#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
#         hrus = res['mappedtile_hrus'].copy()
#         hills = res['mappedtile_hills'].copy().astype(np.int)
#         hrus[hrus<0] = -1
#         hills[hills<0] = -1
#         # hills[hills>10000] = -1
#         cm0 = axes[0].pcolormesh(X, Y, hrus , cmap='Pastel1')
#         cm1 = axes[1].imshow(hills , cmap='Pastel1')
#         # cm0 = axes[1].pcolormesh(X, Y, res['mappedtile_fdir'], cmap='jet', vmin=-1, vmax=1)
#         # axes[0].set_title('{}'.format(mydomain))
#         # axes.set_title('{}'.format(mydomain))
#         # axes[1].set_title(r'tile-by-tile predictions ($n_T = {}$)'.format(res['ntiles']))
#         # axes.set_xlabel('x [km]')
#         # axes.set_ylabel('y [km]')
#         axes[0].set_xlabel('Longitude')
#         axes[0].set_ylabel('Latitude')
#         axes[1].set_xlabel('Longitude')
#         axes[1].set_ylabel('Latitude')
#         cbar0 = fig.colorbar(cm0, ax=axes[0])
#         cbar1 = fig.colorbar(cm1, ax=axes[1])
#         cbar0.set_label(r'Tile')
#         cbar1.set_label(r'Chill')
#         plt.savefig( os.path.join(outfigdir, 'elev_map_{}.png'.format(mydomain)))
#         plt.show()
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
