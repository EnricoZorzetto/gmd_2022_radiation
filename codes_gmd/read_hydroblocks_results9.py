
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
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()

# scenarios = ['10', '25', '100'] # number of HRUs in each simulation
# scenarios = ['100'] # number of HRUs in each simulation
# scenarios = ['sia10', 'sia25', 'sia100'] # ONLY cos(sia) used in the clustering
# scenarios = ['notnorm10', 'notnorm25', 'notnorm100'] # ONLY cos(sia) used in the clustering


datadir_all = os.path.join('..', '..', '..', 'Documents', 'res_hmc_light_p4')

DOMAINS = ['EastAlps']
# DOMAINS = ['EastAlps', 'Peru', 'Nepal']
ndomains = len(DOMAINS)

NHILLS = np.array([10])
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


X, Y = np.meshgrid(all_res[0]['xlon_ea'], all_res[0]['ylat_ea']) # coords equal area in latlon
Xl, Yl = np.meshgrid(all_res[0]['xlon_latlon'], all_res[0]['ylat_latlon']) # coords equal area in latlon


matplotlib.use('Qt5Agg') # dyn show plots
hrus = res['mappedtile_hrus'].copy()
hrus[hrus < 0] = 0

Ct = res['map_tcf']
Vd = res['map_svf']

SS = res['map_sinssina']
SC = res['map_sinscosa']

zmap = res['map_ele']

# do this for elev, all terrain variables, hrus
X1 = X[      100:500,       -500:-100]
Y1 = Y[      100:500,       -500:-100]
hrus = hrus[100:500,       -500:-100]
SS = SS[      100:500,       -500:-100]
SC = SC[      100:500,       -500:-100]
Ct = Ct[      100:500,       -500:-100]
Vd = Vd[      100:500,       -500:-100]

zmap = zmap[      100:500,       -500:-100]

plt.figure(figsize=(5.5, 5))
cm0 = plt.pcolormesh(X1, Y1, hrus, cmap='Pastel1')
plt.ylabel('Latitude [N]')
plt.xlabel('Longitude [E]')
cbar0 = plt.colorbar(cm0)
plt.tight_layout()
# plt.title('tiles')
cbar0.set_label('Tiles')
plt.savefig(os.path.join(outfigdir, 'pastel_tiles_k_{}.png'.format(NHILLS[0])), dpi=100)
plt.show()


plt.figure(figsize=(5.5, 5))
cm0 = plt.pcolormesh(X1, Y1, zmap, cmap='terrain')
plt.ylabel('Latitude [N]')
plt.xlabel('Longitude [E]')
cbar0 = plt.colorbar(cm0)
cbar0.set_label('Elevation [m m.s.l.]')
plt.tight_layout()
plt.savefig(os.path.join(outfigdir, 'zmap_tiles_k_{}.png'.format(NHILLS[0])), dpi=100)
plt.show()

qmin = 0.02
qmax = 0.98
fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (12, 10))

vmin = np.quantile(Ct, qmin)
vmax = np.quantile(Ct, qmax)
cm0 = axes[0,0].pcolormesh(X1, Y1, Ct, cmap='jet', vmin = vmin, vmax = vmax)
cbar0 = fig.colorbar(cm0, ax=axes[0,0], extend='both')
cbar0.set_label(r'$C_t$')

vmin = np.quantile(Vd, qmin)
vmax = np.quantile(Vd, qmax)
cm1 = axes[1,0].pcolormesh(X1, Y1, Vd, cmap='jet', vmin = vmin, vmax = vmax)
cbar1 = fig.colorbar(cm1, ax=axes[1,0], extend='both')
cbar1.set_label(r'$V_d$')

vmin = np.quantile(SS, qmin)
vmax = np.quantile(SS, qmax)
cm2 = axes[0,1].pcolormesh(X1, Y1, SS, cmap='jet', vmin = vmin, vmax = vmax)
cbar2 = fig.colorbar(cm2, ax=axes[0,1], extend='both')
cbar2.set_label(r'$\sin(\theta_s) \sin (\phi_s)$')

vmin = np.quantile(SC, qmin)
vmax = np.quantile(SC, qmax)
cm3 = axes[1,1].pcolormesh(X1, Y1, SC, cmap='jet', vmin = vmin, vmax = vmax)
cbar3 = fig.colorbar(cm3, ax=axes[1,1], extend='both')
cbar3.set_label(r'$\sin(\theta_s) \cos (\phi_s)$')

for i in range(2):
    for j in range(2):
        if j == 0:
            axes[i,j].set_ylabel('Latitude [N]')
        if i == 1:
            axes[i, j].set_xlabel('Longitude [E]')

plt.tight_layout()
plt.savefig(os.path.join(outfigdir, 'terrvars4_tiles_k_{}.png'.format(NHILLS[0])), dpi=100)
plt.show()

TILES = np.unique(hrus)

# TILES = np.array([til for til in list(TILES) if til in [5, 10, 15, 20, 25, 30, 35]])
# TILES = np.array([til for til in list(TILES) if til in [5, 10, 15, 20, 25, 30, 35]])
# TILES = np.flipud(np.unique(hrus))
ntiles = len(TILES)
colors = plt.cm.jet(np.linspace(0,1,ntiles))# Initialize holder for tiles
alphas = np.linspace(0.6, 0.1, ntiles)

# plt.figure()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
for it, tile in enumerate(TILES):

    myct = Ct[hrus == tile]
    myvd = Vd[hrus == tile]
    myss = SS[hrus == tile]
    mysc = SC[hrus == tile]
    axes[0].scatter(myvd, mysc, marker='o', color = colors[it], alpha = alphas[it])
    axes[1].scatter(myct, myss, marker='o', color = colors[it], alpha = alphas[it])

axes[0].set_xlabel(r'$V_d$')
axes[0].set_ylabel(r'$\sin(\theta_s) \cos (\phi_s)$')
axes[1].set_xlabel(r'$C_t$')
axes[1].set_ylabel(r'$\sin(\theta_s) \sin (\phi_s)$')
plt.tight_layout()
plt.savefig(os.path.join(outfigdir, 'scatter_tiles_k_{}.png'.format(NHILLS[0])), dpi=100)
plt.show()


#
# plt.figure()
# for it, tile in enumerate(TILES):
#
#     myct = Ct[hrus == tile]
#     myvd = Vd[hrus == tile]
#     myss = SS[hrus == tile]
#     mysc = SC[hrus == tile]
#
#     # plt.plot(mysc, myss, 'o', color = colors[it])
#     plt.scatter(myct, mysc, marker='o', color = colors[it], alpha=alphas[it])
#
# # plt.xscale('log')
# # plt.xscale('log')
# plt.xlabel(r'$C_t$')
# plt.ylabel(r'$\sin(\theta_s) \sin (\phi_s)$')
# plt.savefig(os.path.join(outfigdir, 'scatter_tiles_2.png'), dpi=100)
# plt.show()
#
