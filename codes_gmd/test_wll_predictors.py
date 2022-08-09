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


# DOMAINS = ['Peru', 'EastAlps', 'Nepal']
# CROP_BUFFERS = [0.1, 0.25]


DOMAINS = ['EastAlps']
# CROP_BUFFERS = [0.3]


# simul_name = 'output_cluster_PP3D_Peru_run2_a35'
simul_name = 'output_cluster_PP3D_EastAlps_run1_a17'
domain_name = simul_name.split('_')[3]

datadir = os.path.join(simul_folder, simul_name)
datadir_out = os.path.join(simul_folder, 'simul_differences')

if not os.path.exists(datadir_out):
    os.makedirs(datadir_out)
datadirm = os.path.join( simul_folder, 'models_{}'.format(domain_name))


adir_prod = simul_name.split('_')[-1]

ndomains = len(DOMAINS)
# nbuffers = len(CROP_BUFFERS)


outfigdir = os.path.join(datadir, '..', 'outfigdir')

if not os.path.exists(datadirm):
    os.makedirs(datadirm)

# East Alps variables

load_existing_models = False
if load_existing_models:
    prefit_models_domain = 'EastAlps'
    # prefit_models_domain = 'Peru'
    prefit_models_adir = 0.5
    datadirm_prefit = os.path.join( simul_folder,
                'models_{}'.format(prefit_models_domain))







# First, read the RMC results and save the differences 3D - PP fluxes
# for various averaging tile sizes
do_average = True
crop_buffer = 0.25
FTOAnorm=1361.0
#  size [km]  0.5  1   3   5   10



# variables to study
# LABELS = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup',
#           'frdirn', 'frdifn', 'fcoupn']

# AVEBLOCKS =   [6, 12, 32, 55, 110]
AVEBLOCKS =   [110]
naveblocks = len(AVEBLOCKS)



LABELS = ['fdir', 'fdif',
          'frdirn', 'frdifn', 'fcoupn']
nlabels = len(LABELS)

# nblocks = len(AVEBLOCKS)


# LABELS = ['fcoupn']
################################################################################
#################  TO READ THE DIFFERENCE FROM MC OUTPUT FILES #################
read_original_output = False
if read_original_output:
    for ib in range(naveblocks):
        aveblock = AVEBLOCKS[ib]
        dem.read_PP_3D_differences(datadir=datadir,
                                   outdir=datadirm,
                                   do_average=do_average,
                                   aveblock=aveblock,
                                   crop_buffer=crop_buffer,
                                   FTOAnorm=FTOAnorm)
################################################################################


for ib in range(naveblocks):
    aveblock = AVEBLOCKS[ib]


    mlr_savedir = os.path.join(datadirm,
        'wll_models_ave_{}'.format(aveblock))
    if not os.path.exists(mlr_savedir):
        os.makedirs(mlr_savedir)

    # result obtained averaging over the 2 albedos of current simulation
    # ds_ave = xr.open_dataset( os.path.join(datadirm,
    #  'train_test_data_size_{}_adir_ave_{}.nc'.format(
    #                                     aveblock, adir_prod)))

    # albedo-by-albedo results
    ds_sin = xr.open_dataset( os.path.join(datadirm,
         'train_test_data_size_{}_adir_singles_{}.nc'.format(
                                        aveblock, adir_prod)))

    ADIRs = ds_sin['ADIRs'].values



    # COSZs = ds_ave['COSZs'][:].values
    COSZs = ds_sin['COSZs'][:].values
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


            # train and test predictive models for each label
            # for current cosz and adir combination
            for il, label in enumerate(LABELS):

                dfvars2 = dfvars.copy()
                # --------------------------------------------------------------
                # change variables based on label and cosz values
                if label in ['frdirn', 'frdifn'] or (mycosz > 0.99 and label == 'fdir'):
                    _ = dfvars2.pop('sian')
                # else:
                #     dfvars = dfvars_gen

                # if mycosz > 0.99 and label == 'fdir':
                #     print('Removing SIA from predictors in case cosz = 1')
                #     _ = dfvars.pop('sian')
                # --------------------------------------------------------------


                # ------------ divide data in training and testing -------------
                print("preparing df for training and testing models")
                dfred = {var:dfvars2[var] for var in dfvars2.keys()}
                dfred[label] = dfflux[label] # add current label to the dict/df
                dfred = pd.DataFrame(dfred)

                train_dataset = dfred
                # test_dataset = dfred
                train_features = dfred.copy()
                train_labels = train_features.pop(label)
                # test_features = train_features
                # test_labels = train_labels


                # scal = StandardScaler()
                # Xrf_train = scal.fit_transform(train_features)
                Xrf_train = train_features

                # if not load_existing_models:

                mlr_savename = 'mlr_model_{}_cosz_{}_adir_{}'.format(
                    label, mycosz, myadir)
                rfr_savename = 'rfr_model_{}_cosz_{}_adir_{}'.format(
                    label, mycosz, myadir)


                # --------------- fit MULTIPLE LINEAR REGRESSION ---------------
                print('Training multiple linear regression')
                mlr_model = LinearRegression()
                mlr_model.fit(Xrf_train, train_labels)

                # get variable importance
                # NOTE - THIS ASSUMES THAT VARIABLES HAVE BEEN NORMALIZED BEFORE
                importance = mlr_model.coef_
                for i, v in enumerate(importance):
                    print('Feature: %0d, Score: %.5f' % (i, v))

                with open(os.path.join(mlr_savedir,
                        '{}.pickle'.format(mlr_savename)), 'wb') as pklf:
                    pickle.dump(mlr_model, pklf)

                # read saved models and do predictions
                with open(os.path.join(mlr_savedir,
                         '{}.pickle'.format(mlr_savename)), 'rb') as pklf:
                    mlr_model = pickle.load(pklf)


                ymlr_pred = mlr_model.predict(Xrf_train)
                # --------------------------------------------------------------

                # -------------- compare with WLL parameterization -------------

                SSres_mlr = np.sum((train_labels - ymlr_pred)**2)
                SStot = np.sum((train_labels - np.mean(train_labels))**2)
                R2mlr = 1 - SSres_mlr/SStot

                if label not in ['fcoup', 'fcoupn']:
                    ylee_pred = dem.lee_model_predict(train_dataset,
                                label=label, cosz=mycosz, albedo=myadir)
                    SSres_wll = np.sum((train_labels - ymlr_pred)**2)
                    R2wll = 1 - SSres_wll/SStot

                plt.figure(figsize=(10, 10))
                if label not in ['fcoup', 'fcoupn']:
                    plt.plot(train_labels, ylee_pred, 'or',
                         label = r'WLL $r^2 = {:.2f}$'.format(R2wll))
                plt.plot(train_labels, ymlr_pred, 'ob',
                         label = r'MLR $r^2 = {:.2f}$'.format(R2mlr))
                # plt.plot(train_labels, yrf_pred, 'og',
                #          label = r'RF $R^2 = {:.2f}$'.format(R2rf))
                plt.plot(ymlr_pred, ymlr_pred, 'k')
                plt.title('pred {} flux, cosz = {}.'.format(label, mycosz))
                plt.xlabel('{} - Monte Carlo Simulation'.format(label))
                plt.ylabel('{} - parameterization prediction'.format(label))
                lee_savename = 'lee_model_{}_cosz_{}_adir_{}.png'.format(
                                        label, mycosz,  myadir)


                plt.legend()
                plt.savefig( os.path.join(mlr_savedir, lee_savename))
                plt.close()
