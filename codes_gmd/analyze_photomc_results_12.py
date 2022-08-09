import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import demfuncs as dem
from topocalc.viewf import viewf, gradient_d8
import pickle
import string
# import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import itertools
from sklearn.linear_model import LinearRegression


import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()



################################################################################
# simul_name = 'output_cluster_PP3D_EastAlps_run2_a35'
# simul_name = 'output_cluster_PP3D_EastAlps_run1_a17'
# simul_name = 'output_cluster_PP3D_EastAlps_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run2_a35'
# simul_folder = os.path.join('//', 'home', 'enrico', 'Documents','dem_datasets') #  DUKE LAPTOP
simul_folder = os.path.join('//', 'Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP

outfigdir = os.path.join('//', 'home', 'enrico', 'Documents',
                               'dem_datasets', 'outfigdir')


# folder with the original RMC model output

# This is domain used to load test data
# and used for training only if load_existing_models = False


# DOMAINS = ['Nepal']
DOMAINS = ['EastAlps']
# DOMAINS = ['Peru']
# DOMAINS = ['Peru', 'EastAlps', 'Nepal']
# CROP_BUFFERS = [0.1, 0.25]
CROP_BUFFERS = [0.25]
# CROP_BUFFERS = [0.2]

ndomains = len(DOMAINS)
nbuffers = len(CROP_BUFFERS)


# domain_name = simul_name.split('_')[3]
# simul_name = 'output_cluster_PP3D_Peru_run2_a35'

# datadir = os.path.join(simul_folder, simul_name)

datadir_diffs = os.path.join(simul_folder, 'simul_differences')
datadir_models = os.path.join(simul_folder, 'trained_models')
datadir_crossval = os.path.join(simul_folder, 'crossval_models')

# if not os.path.exists(datadir_out):
#     os.makedirs(datadir_out)

if not os.path.exists(datadir_diffs):
    os.makedirs(datadir_diffs)

if not os.path.exists(datadir_models):
    os.makedirs(datadir_models)

# if not os.path.exists(crossval_models):
#     os.makedirs(crossval_models)

if not os.path.exists(datadir_crossval):
    os.makedirs(datadir_crossval)

# datadirm = os.path.join( simul_folder, 'models_{}'.format(domain_name))


# adir_prod = simul_name.split('_')[-1]
# adir_prod = 'a17' # part of file name indicating available albedo values
adir_prod = 'a35' # part of file name indicating available albedo values




# outfigdir = os.path.join(datadir, '..', 'outfigdir')
#
# if not os.path.exists(datadirm):
#     os.makedirs(datadirm)

# East Alps variables


test_only = True
make_plots = True
# test_only = False
# make_plots = True
# load_existing_models = False
if test_only:
    # prefit_models_domain = 'EastAlps'
    prefit_models_domain = 'Peru'
    # prefit_models_domain = 'Nepal'
    # prefit_models_domain = 'Peru'
    prefit_models_adir = 0.3
else:
    prefit_models_domain = None
    prefit_models_adir = np.nan







# First, read the RMC results and save the differences 3D - PP fluxes
# for various averaging tile sizes
do_average = True
# crop_buffer = 0.20
FTOAnorm = 1361.0 # * mycosz
#  size [km]  0.5  1   3   5   10



# variables to study
# LABELS = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup',
#           'frdirn', 'frdifn', 'fcoupn']

# AVEBLOCKS =   [6, 12, 32, 55, 110]
AVEBLOCKS =   [110]
naveblocks = len(AVEBLOCKS)


# AVEBLOCKS =   [55, 110]

LABELS = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
nlabels = len(LABELS)


# save netcdf as test
file_res = os.path.join(datadir_models,
                        'R2res_test_trained_{}_{}.nc'.format(
                            prefit_models_domain, prefit_models_adir))
# modeldata.to_netcdf(file_res)


ds = xr.open_dataset(file_res)

# fluxes = ds['FLUXES'].values
COSZs = ds['COSZs'].values
ncosz = len(COSZs)

LABELS = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
LABEL_SYMB = [r'$f_{dir}$', r'$f_{dif}$', r'$f_{rdir}$', r'$f_{rdif}$', r'$f_{coup}$', ]
nlabels = len(LABELS)
# matplotlib.use('Qt5Agg') # dyn show plots
# plt.figure()

# iadir = 0
iadir = 1

pad = 7
figsize = 3.0

fig, axes = plt.subplots(nrows=ncosz, ncols=nlabels, figsize = (1.8*figsize*nlabels, figsize*ncosz))

for ic, mycosz in enumerate(COSZs):
    for il, mylabel in enumerate(LABELS):

        axes[ic, il].plot(ds['AVEBLOCKS'].values*90/1000, ds['R2'][dict(idom=0,ibuf=0,iadir=iadir,icosz=ic,imod=0,ilab=il)].values,       linewidth = 2.0, color = 'green', label = ds['DOMAINS'][0].values)
        axes[ic, il].plot(ds['AVEBLOCKS'].values*90/1000, ds['R2'][dict(idom=1,ibuf=0,iadir=iadir,icosz=ic,imod=0,ilab=il)].values,       linewidth = 2.0, color = 'red', label = ds['DOMAINS'][1].values)
        # axes[ic, il].plot(ds['AVEBLOCKS'].values*90/1000, ds['R2'][dict(idom=2,ibuf=0,iadir=iadir,icosz=ic,imod=0,ilab=il)].values,       linewidth = 2.0, color = 'blue', label = ds['DOMAINS'][2].values)
        axes[ic, il].plot(ds['AVEBLOCKS'].values*90/1000, ds['R2'][dict(idom=0,ibuf=0,iadir=iadir,icosz=ic,imod=1,ilab=il)].values, '--', linewidth = 2.0, color = 'green')
        axes[ic, il].plot(ds['AVEBLOCKS'].values*90/1000, ds['R2'][dict(idom=1,ibuf=0,iadir=iadir,icosz=ic,imod=1,ilab=il)].values, '--', linewidth = 2.0, color = 'red'  )
        # axes[ic, il].plot(ds['AVEBLOCKS'].values*90/1000, ds['R2'][dict(idom=2,ibuf=0,iadir=iadir,icosz=ic,imod=1,ilab=il)].values, '--', linewidth = 2.0, color = 'blue' )
        # axes[ic, il].plot(ds['AVEBLOCKS'].values, ds['R2'][dict(idom=0,ibuf=0,iadir=0,icosz=ic,imod=2,ilab=il)].values, '-o', color = 'green')
        # axes[ic, il].plot(ds['AVEBLOCKS'].values, ds['R2'][dict(idom=1,ibuf=0,iadir=0,icosz=ic,imod=2,ilab=il)].values, '-o', color = 'red'  )
        # axes[ic, il].plot(ds['AVEBLOCKS'].values, ds['R2'][dict(idom=2,ibuf=0,iadir=0,icosz=ic,imod=2,ilab=il)].values, '-o', color = 'blue' )


        axes[ic,il].grid(True)
        # axes[ic, il].set_xlabel('scale [npixels]')
        # axes[ic, il].set_ylabel(r'$R^2$')
        if ic == 0 and il == 0:
            # axes[ic, il].legend()
            # axes[ic, il].legend(loc='upper right', ncol=5, bbox_to_anchor=(4.0, 1.85))  # legend only on first plot
            axes[ic, il].legend(loc='lower right')
        axes[ic, il].set_ylim([0,1])

        if il == 0:
            axes[ic, il].annotate( r'$\mu_0$ = {}'.format(COSZs[ic]),
                                xy=(0, 0.5), xytext=(
                    -axes[ic, il].yaxis.labelpad - pad, 0),
                                xycoords=axes[ic, il].yaxis.label,
                                textcoords='offset points',
                                size='large', ha='right', va='center')
        if ic == 0:
            axes[ic, il].set_title('{}'.format(LABEL_SYMB[il]))
        # if ic == ncosz - 1:
        #     axes[ic, il].set_xlabel(r'{}'.format(LABELS[il]))

        if ic == ncosz - 1:
            axes[ic, il].set_xlabel(r'scale [km]')
        else:
            axes[ic, il].xaxis.set_ticklabels([])
            # axes[ic, il].set_xlabel([])

        if il == 0:
            axes[ic, il].set_ylabel(r'$R^2$')
        else:
            axes[ic, il].yaxis.set_ticklabels([])
            # axes[ic, il].set_ylabel([])

    # plt.plot(modeldata['R2'][dict(idom=1,ibuf=0,iadir=0,icosz=2,imod=0,ilab=0)].values)
plt.tight_layout()
plt.savefig(os.path.join(outfigdir, 'R2_res_trained_{}'.format(prefit_models_domain)), dpi = 300)
plt.show()
