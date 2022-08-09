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
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()



################################################################################
# simul_name = 'output_cluster_PP3D_EastAlps_run2_a35'
# simul_name = 'output_cluster_PP3D_EastAlps_run1_a17'
# simul_name = 'output_cluster_PP3D_EastAlps_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run2_a35'
# simul_folder = os.path.join('//', 'home', 'enrico', 'Documents','dem_datasets')
simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP

# folder with the original RMC model output

# This is domain used to load test data
# and used for training only if load_existing_models = False


# DOMAINS = ['Nepal']
DOMAINS = ['EastAlps']
# DOMAINS = ['Peru']
# DOMAINS = ['Peru', 'EastAlps']
# CROP_BUFFERS = [0.1, 0.25]
# CROP_BUFFERS = [0.1]
CROP_BUFFERS = [0.10]

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


test_only = False
make_plots = True
# load_existing_models = False
if test_only:
    # prefit_models_domain = 'EastAlps'
    # prefit_models_domain = 'EastAlps'
    prefit_models_domain = 'Nepal'
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

AVEBLOCKS =   [6, 12, 32, 55, 110]
# AVEBLOCKS =   [6]
naveblocks = len(AVEBLOCKS)


# AVEBLOCKS =   [55, 110]

LABELS = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
nlabels = len(LABELS)

# nblocks = len(AVEBLOCKS)



# for plotting purposes only::
# plot maps of high-res (before cropping and averaging)
# of terrain variables and of rad flux differences 3D-PP
# and relative scatter plots

plot_original_fields = False
if plot_original_fields:
    for id, mydomain in enumerate(DOMAINS):

        simul_name = 'output_cluster_PP3D_{}_run1_{}' \
            .format(mydomain, adir_prod)

        datadir = os.path.join(simul_folder, simul_name)

        outfigdir_fields = os.path.join(datadir_diffs, 'plots_{}'.format(mydomain))

        if not os.path.exists(outfigdir_fields):
            os.makedirs(outfigdir_fields)

        dem.read_PP_3D_differences(datadir=datadir,
                                   outdir=None,
                                   do_average=True,
                                   aveblock= 6,
                                   crop_buffer= 0.20,
                                   do_plots = True,
                                   save_data=False,
                                   outfigdir=outfigdir_fields,
                                   FTOAnorm=FTOAnorm)

# exit()


# LABELS = ['fcoupn']
################################################################################
#################  TO READ THE DIFFERENCE FROM MC OUTPUT FILES #################
read_original_output = False # to be done only once
# read_original_output = False # to be done only once
if read_original_output:
    for id, mydomain in enumerate(DOMAINS):
        for ibu, mybuffer in enumerate(CROP_BUFFERS):
            for ibl, myaveblock in enumerate(AVEBLOCKS):

                # simul_name = 'output_cluster_PP3D_{}_run1_{}'\
                #                    .format(mydomain, adir_prod)
                simul_name = 'output_cluster_PP3D_{}_run1_{}'\
                                    .format(mydomain, adir_prod)

                datadir = os.path.join(simul_folder, simul_name)

                outdiffdif = os.path.join(datadir_diffs,
                            'domain_{}_buffer_{}'.format(mydomain, mybuffer))

                # for debug purposes only
                # dem.read_PP_3D_differences(datadir=datadir, outdir=None,
                #                        do_average=False,
                #                        aveblock=myaveblock, crop_buffer=mybuffer,
                #                        FTOAnorm=FTOAnorm, do_debug=True)

                if not os.path.exists(outdiffdif):
                    os.makedirs(outdiffdif)

                dem.read_PP_3D_differences(datadir=datadir,
                                           outdir=outdiffdif,
                                           do_average=do_average,
                                           aveblock=myaveblock,
                                           crop_buffer=mybuffer,
                                           do_plots=False,
                                           FTOAnorm=FTOAnorm)


################################################################################




# train models
def init_r2_dataset():
    nmodels = 3
    modeldata = xr.Dataset(
        {
            "R2": (
                ("idom", "ibuf", "iave", "iadir", "icosz", "imod", "ilab"),
                np.zeros((ndomains,  nbuffers, naveblocks,
                          nadir, ncosz, nmodels, nlabels),
                dtype=np.float32)),

            "MODELS": ("imod", ['MLR', 'RFR', 'WLL']),
            "DOMAINS": ("idom", DOMAINS),
            "CROP_BUFFERS": ("ibuf", CROP_BUFFERS),
            "AVEBLOCKS": ("iave", AVEBLOCKS),
            "ADIRs": ("iadir", ADIRs),
            "LABELS": ("ilab", LABELS),
            "COSZs": ("icosz", COSZs)
        },
        coords={
            "imod": np.arange(nmodels),
            "ilab": np.arange(nlabels),
            "idom": np.arange(ndomains),
            "ibuf": np.arange(nbuffers),
            "iave": np.arange(naveblocks),
            "iadir": np.arange(nadir),
            "icosz": np.arange(ncosz)
        }
        # attrs={
        #     "coup_frac_pp": coup_frac_pp
        # }
    )
    return modeldata



for id, mydomain in enumerate(DOMAINS):
    for ibu, mybuffer in enumerate(CROP_BUFFERS):
        for ibl, myaveblock in enumerate(AVEBLOCKS):
            # print(id, ibu, ibl)


            mydatadir_diffs = os.path.join(datadir_diffs,
                                           'domain_{}_buffer_{}'.format(mydomain, mybuffer))

            model_savedir = os.path.join(datadir_models,
                                         'domain_{}_buffer_{}'.format(mydomain, mybuffer),
                                         'models_ave_{}'.format(myaveblock) )

            if not test_only:
                if not os.path.exists(model_savedir):
                    os.makedirs(model_savedir)
                test_savedir = None

            else:
                if not os.path.exists(model_savedir):
                    raise Exception('Error: the folder with saved '
                                    'models must exist!')

                test_savedir = os.path.join(datadir_crossval,
                        'domain_{}_buffer_{}'.format(mydomain, mybuffer),
                        'cv_train_{}_{}_test_{}_aveblock_{}'.format(
                        prefit_models_domain, prefit_models_adir,
                        mydomain, myaveblock
                        )
                        )

                if not os.path.exists(test_savedir):
                    os.makedirs(test_savedir)

            # read data [simulation results]
            # result obtained averaging over the 2 albedos of current simulation
            ds_ave = xr.open_dataset( os.path.join(mydatadir_diffs,
                        'train_test_data_size_{}_adir_ave_{}.nc'.format(
                                                myaveblock, adir_prod)))

            # albedo-by-albedo results
            ds_sin = xr.open_dataset( os.path.join(mydatadir_diffs,
                        'train_test_data_size_{}_adir_singles_{}.nc'.format(
                                                myaveblock, adir_prod)))

            # plt.figure()
            # plt.imshow(ds_sin['fdir'][:, 0,0])
            # plt.show()

            ADIRs = ds_sin['ADIRs'].values
            # COSZs = ds_ave['COSZs'][:].values
            COSZs = ds_sin['COSZs'][:].values
            nadir = np.size(ADIRs)
            ncosz = np.size(COSZs)

            # initialize xarray dataset
            if id==0 and ibu==0 and ibl==0:
                modeldata = init_r2_dataset()

            for ic, mycosz in enumerate(COSZs):
                for ia, myadir in enumerate(ADIRs):
                    dfflux = pd.DataFrame({
                        'fdir':   ds_sin['fdir'][:,   ic, ia].values,
                        'fdif':   ds_sin['fdif'][:,   ic, ia].values,
                        'frdir':  ds_sin['frdir'][:,  ic, ia].values,
                        'frdif':  ds_sin['frdif'][:,  ic, ia].values,
                        'fcoup':  ds_sin['fcoup'][:,  ic, ia].values,
                        'frdirn': ds_sin['frdirn'][:, ic, ia].values,
                        'frdifn': ds_sin['frdifn'][:, ic, ia].values,
                        'fcoupn': ds_sin['fcoupn'][:, ic, ia].values
                    })

                    # predictor variables - use dimensionless elevation only
                    # particularly difficult to extrapolate
                    # ds_sin['ele'][:].values = ds_sin['ele'][:].values - np.mean(
                    #     ds_sin['ele'][:].values) / np.std(ds_sin['ele'][:].values)
                    # ds_sin['sde'][:].values = ds_sin['sde'][:].values - np.mean(
                    #     ds_sin['sde'][:].values) / np.std(ds_sin['sde'][:].values)
                    dfvars = pd.DataFrame({
                        # 'ele': ds_sin['ele'][:].values,
                        # 'sde':  ds_sin['sde'][:].values,
                        'svfn': ds_sin['svfn'][:].values,
                        'tcfn': ds_sin['tcfn'][:].values,
                        'sian': ds_sin['sian'][:, ic].values
                    })


                    R2 = dem.train_test_models(
                                dfflux=dfflux, dfvars=dfvars,
                                LABELS=LABELS,
                                mycosz=mycosz,
                                myadir=myadir, myaveblock=myaveblock,
                                test_only=test_only, make_plot=make_plots,
                                prefit_models_adir=prefit_models_adir,
                                modeldir=model_savedir,
                                testdir=test_savedir)

                    # ("idom", "ibuf", "iave", "iadir", "icosz", "imod", "ilab"),
                    # print(R2)
                    print(id, ibu, ibl)
                    # print(modeldata['R2'][id, ibu, ibl, ia, ic, :, :].values.shape)
                    # print(R2.shape)
                    modeldata['R2'][id, ibu, ibl, ia, ic, :, :] = R2
                    # print(modeldata['R2'][id, ibu, ibl, ia, ic, :, :].values)
                    # print(np.size(modeldata['R2'][id, ibu, ibl, ia, ic, :, :].values))
                    # modeldata['R2'][:, 0, :, 0, 0, 0, 0].values
                    # print('Equal')


                    # save R2 values in a netcdf file

                    # ds.to_netcdf(file_res)
                    # ds_ia.to_netcdf(file_res_ia)

if test_only:
    # save netcdf as test
    file_res = os.path.join(datadir_models,
            'R2res_test_trained_{}_{}.nc'.format(
            prefit_models_domain, prefit_models_adir))
    modeldata.to_netcdf(file_res)
else:
    # save netcdf as train
    file_res = os.path.join(datadir_models,
            'R2res_train.nc')
    modeldata.to_netcdf(file_res)


                    # ave block (pos 2) is what determines the 0
                    # modeldata['R2'][:, 0, 4, :, :, :, :].values
                    # modeldata['R2']




matplotlib.use('Qt5Agg') # dyn show plots
plt.figure()
# # plt.plot(modeldata['AVEBLOCKS'], modeldata['R2'].sel(['EastAlps', 0.25, :, 0.3, 0.5])
# plt.plot(modeldata['AVEBLOCKS'], modeldata['R2'][dict(imod='EastAlps',
#                                                       ibuf=0.25,
#                                                       iadir=0.3,
#                                                       icosz=0.5,
#                                                       imod='MLR',
#                                                       ilab='fdir')])


plt.plot(modeldata['AVEBLOCKS'].values, modeldata['R2'][dict(idom=1,
                                                      ibuf=0,
                                                      iadir=0,
                                                      icosz=2,
                                                      imod=0,
                                                      ilab=0)].values)

plt.plot(modeldata['R2'][dict(idom=1,ibuf=0,iadir=0,icosz=2,imod=0,ilab=0)].values)

plt.show()


# modeldata['AVEBLOCKS'].values
# modeldata['R2'][dict(idom=1,ibuf=0,iadir=0,icosz=2,imod=0,ilab=0)].values
# modeldata['R2'][dict(idom=0,ibuf=0,iadir=0,icosz=2,imod=1,ilab=0)].values
#
# np.shape(modeldata['R2'])
#

            # optionally, to modifiy copuple flux::

            # ff1_test =   (1
            #             + ds_sin['fdir'][:, ic, ia_test].values* test_adir
            #             + ds_sin['fdif'][:, ic, ia_test].values)
            # ff1_train = (1
            #             + ds_sin['fdir'][:, ic, ia_train].values * train_adir
            #             + ds_sin['fdif'][:, ic, ia_train].values)
            #
            # coup_frac_test = ds_sin['coup_frac_pp'].values[ia_test]
            # coup_frac_train = ds_sin['coup_frac_pp'].values[ia_train]
            # coup_frac_test_3d = ds_sin['coup_frac_3d'].values[ia_test]
            # coup_frac_train_3d = ds_sin['coup_frac_3d'].values[ia_train]

            # matplotlib.use('Qt5Agg') # dyn show plots
            # sums = tcfo + svfo
            # plt.figure()
            # plt.hist(sums)
            # plt.show()

            ####################################################################
            #### find a correction factor to normalize the coupled flux ####
            #### and obtain albedo - independent predictions ---------- ####
            # tcf0 = ds_sin['tcf0'][:].values
            # svf0 = ds_sin['svf0'][:].values
            # svfn = ds_sin['svfn'][:].values
            # tcfn = ds_sin['tcfn'][:].values
            # alphas_rho_test = coup_frac_test / (1 + coup_frac_test)
            # alphas_rho_train = coup_frac_train / (1 + coup_frac_train)
            # A3D_test = alphas_rho_test * (1 - np.mean(tcf0)) / (
            #         1 - alphas_rho_test * (1 - np.mean(tcf0)))
            # A3D_train = alphas_rho_train * (1 - np.mean(tcf0)) / (
            #             1 - alphas_rho_train * (1 - np.mean(tcf0)))

            # corrfact_test =  (1-np.mean(tcf0))*(test_adir  * np.mean(tcf0**2))  / (1 - test_adir  *np.mean(tcf0) )
            # corrfact_train = (1-np.mean(tcf0))*(train_adir * np.mean(tcf0**2)) / (1 - train_adir * np.mean(tcf0) )

            # corrfact_train = 1
            # corrfact_test =  1
            #
            # # corrfact_test =  (test_adir )**0.5  / (1 - test_adir  *np.mean(tcf0) )
            # # corrfact_train = (train_adir)**0.5 / (1 - train_adir * np.mean(tcf0) )
            #
            # # dfflux_testing_albedo['fcoupn'] = dfflux_testing_albedo['fcoupn'] / corrfact_test
            # # dfflux['fcoupn'] = dfflux['fcoupn'] / corrfact_train
            #
            # dfflux_testing_albedo['fcoupn'] = dfflux_testing_albedo['fcoupn'] / corrfact_test
            # dfflux['fcoupn'] = dfflux['fcoupn'] / corrfact_train
            ####################################################################


# save netcdf as test
file_res = os.path.join(datadir_models,
                        'R2res_test_trained_{}_{}.nc'.format(
                            prefit_models_domain, prefit_models_adir))
# modeldata.to_netcdf(file_res)


# ds = xr.open_dataset(file_res)



# matplotlib.use('Qt5Agg') # dyn show plots
# plt.figure()
#
#
# plt.plot(ds['AVEBLOCKS'].values, ds['R2'][dict(idom=1,
#                                                              ibuf=0,
#                                                              iadir=0,
#                                                              icosz=2,
#                                                              imod=0,
#                                                              ilab=0)].values)
#
# # plt.plot(modeldata['R2'][dict(idom=1,ibuf=0,iadir=0,icosz=2,imod=0,ilab=0)].values)
# plt.show()
