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


import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()



################################################################################
# simul_name = 'output_cluster_PP3D_EastAlps_run2_a35'
# simul_name = 'output_cluster_PP3D_EastAlps_run1_a17'
simul_name = 'output_cluster_PP3D_EastAlps_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run1_a35'
# simul_name = 'output_cluster_PP3D_Peru_run1_a35'
# simul_folder = os.path.join('//', 'home', 'enrico', 'Documents','dem_datasets')
# simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP
simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')   # PRINCETON LAPTOP

# folder with the original RMC model output
datadir = os.path.join(simul_folder, simul_name)

domain_name = simul_name.split('_')[3]
adir_prod = simul_name.split('_')[-1]

model_name = 'EastAlps'
prefit_model_folder = os.path.join( simul_folder, 'models_{}'.format(model_name))


datadirm = os.path.join( simul_folder, 'models_{}'.format(domain_name))


outfigdir = os.path.join(datadir, '..', 'outfigdir')

if not os.path.exists(datadirm):
    os.makedirs(datadirm)




# First, read the RMC results and save the differences 3D - PP fluxes
# for various averaging tile sizes
do_average = True
crop_buffer = 0.25
FTOAnorm=1361.0
#  size [km]  0.5  1   3   5   10

cross_valid = False
train_frac = 0.99 # fraction of dataset to be used for model training (if cross_valid)


# variables to study
# LABELS = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup',
#           'frdirn', 'frdifn', 'fcoupn']

# AVEBLOCKS =   [6, 12, 32, 55, 110]
AVEBLOCKS =   [6]


# AVEBLOCKS =   [55, 110]

LABELS = ['fdir', 'fdif',
          'frdirn', 'frdifn', 'fcoupn']

# nblocks = len(AVEBLOCKS)


# LABELS = ['fcoupn']
################################################################################
#################  TO READ THE DIFFERENCE FROM MC OUTPUT FILES #################
read_original_output = False
naveblocks = len(AVEBLOCKS)
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

    # init folder with result for current aveblock
    mlr_savedir = os.path.join(datadirm,
        'mlr_models_ave_{}'.format(aveblock))
    if not os.path.exists(mlr_savedir):
        os.makedirs(mlr_savedir)

    # init folder with result for current aveblock
    mlr_readdir = os.path.join(prefit_model_folder,
                               'mlr_models_ave_{}'.format(aveblock))
    # if not os.path.exists(mlr_savedir):
    #     os.makedirs(mlr_savedir)

    # result obtained averaging over the 2 albedos of current simulation
    ds_ave = xr.open_dataset( os.path.join(datadirm,
     'train_test_data_size_{}_adir_ave_{}.nc'.format(
                                        aveblock, adir_prod)))

    # albedo-by-albedo results
    ds_sin = xr.open_dataset( os.path.join(datadirm,
         'train_test_data_size_{}_adir_singles_{}.nc'.format(
                                        aveblock, adir_prod)))


    COSZs = ds_ave['COSZs'][:].values
    for ic, mycosz in enumerate(COSZs):

        # to test and train the mode lusing different ground reflectivities
        ADIRs = ds_sin['ADIRs']

        # to check tyhe correct normalization of erflected fluxes::
        ia_test = 1 # test model on the first albedo simulation
        ia_train = 0 # and test using the second
        test_adir = ADIRs.values[ia_test]
        train_adir = ADIRs.values[ia_train]

        dfflux_testing_albedo = pd.DataFrame({
            'fdir':   ds_sin['fdir'][:, ic,   ia_test].values,
            'fdif':   ds_sin['fdif'][:, ic,   ia_test].values,
            'frdir':  ds_sin['frdir'][:, ic,  ia_test].values,
            'frdif':  ds_sin['frdif'][:, ic,  ia_test].values,
            'fcoup':  ds_sin['fcoup'][:, ic,  ia_test].values,
            'frdirn': ds_sin['frdirn'][:, ic, ia_test].values,
            'frdifn': ds_sin['frdifn'][:, ic, ia_test].values,
            'fcoupn': ds_sin['fcoupn'][:, ic, ia_test].values
        })

        dfflux = pd.DataFrame({
            'fdir':   ds_sin['fdir'][:,   ic, ia_train].values,
            'fdif':   ds_sin['fdif'][:,   ic, ia_train].values,
            'frdir':  ds_sin['frdir'][:,  ic, ia_train].values,
            'frdif':  ds_sin['frdif'][:,  ic, ia_train].values,
            'fcoup':  ds_sin['fcoup'][:,  ic, ia_train].values,
            'frdirn': ds_sin['frdirn'][:, ic, ia_train].values,
            'frdifn': ds_sin['frdifn'][:, ic, ia_train].values,
            'fcoupn': ds_sin['fcoupn'][:, ic, ia_train].values
        })

        # predictor variables
        dfvars = pd.DataFrame({
            'ele':  ds_sin['ele'][:].values,
            # 'sde':  ds_sin['sde'][:].values,
            'sian': ds_sin['sian'][:, ic].values,
            'svfn': ds_sin['svfn'][:].values,
            'tcfn': ds_sin['tcfn'][:].values
        })



        # ff1_test =   (1
        #                 + ds_sin['fdir'][:, ic, ia_test].values* test_adir
        #                 + ds_sin['fdif'][:, ic, ia_test].values)
        # ff1_train = (1
        #                 + ds_sin['fdir'][:, ic, ia_train].values * train_adir
        #                 + ds_sin['fdif'][:, ic, ia_train].values)
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

        ########################################################################
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
        ########################################################################


        # train and test predictive models for each label
        for il, label in enumerate(LABELS):

            if mycosz > 0.99 and label == 'fdir':
                print('Removing SIA from predictors in case cosz = 1')
                _ = dfvars.pop('sian')

            # ------------ divide data in training and testing -----------------
            print("preparing df for training and testing models")
            dfred = {var:dfvars[var] for var in dfvars.keys()}
            dfred[label] = dfflux[label] # add current label to the dict / df
            dfred = pd.DataFrame(dfred)

            # do the same for the independent albedo
            dfred_testing_albedo = {var:dfvars[var] for var in dfvars.keys()}
            dfred_testing_albedo[label] = dfflux_testing_albedo[label] # add current label to the dict / df
            dfred_testing_albedo = pd.DataFrame(dfred_testing_albedo)



            # Xrf = np.array(dfvars) # predictors
            # yrf = dfflux[label].values # labels
            # yrf_IndepAlbedo = dfflux_testing_albedo[label].values # 2nd alb lab
            #
            # # perform the same train-test split for the 2 albedo values
            # # randnum = random.randint(0, 10000)
            # randnum = 0
            # Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(
            #                        Xrf, yrf, test_size= 1-train_frac,
            #                                  random_state=randnum)
            # _, _, _, yrf_test_IndepAlbedo = train_test_split(
            #     Xrf, yrf_IndepAlbedo,  test_size= 1-train_frac,
            #                            random_state=randnum)
            #
            # print("train predictor shape = {}".format(Xrf_train.shape))
            # print("test perdictor shape = {}".format(Xrf_test.shape))
            # ------------------------------------------------------------------


            # better to use pandas instead:::
            # random_state = 0
            if cross_valid:
                train_dataset = dfred.sample(frac=train_frac, random_state=0)
                test_dataset = dfred.drop(train_dataset.index)
                train_features = train_dataset.copy()
                test_features = test_dataset.copy()
                train_labels = train_features.pop(label)
                test_labels = test_features.pop(label)


                train_dataset_testalbedo = dfred_testing_albedo.sample(frac=train_frac, random_state=0)
                test_dataset_testalbedo = dfred_testing_albedo.drop(train_dataset_testalbedo.index)

                test_features_testalbedo = test_dataset_testalbedo.copy()
                test_labels_testalbedo = test_features_testalbedo.pop(label)

            else:
                train_dataset = dfred
                test_dataset = dfred
                train_features = dfred.copy()
                train_labels = train_features.pop(label)
                test_features = train_features
                test_labels = train_labels

                train_dataset_testalbedo = dfred_testing_albedo
                test_dataset_testalbedo = dfred_testing_albedo
                train_features_testalbedo = dfred_testing_albedo.copy()
                train_labels_testalbedo = train_features_testalbedo.pop(label)
                test_features_testalbedo = train_features_testalbedo
                test_labels_testalbedo = train_labels_testalbedo

            # all_labels_IndepAlbedo = dfflux_testing_albedo[label].values # 2nd alb lab
            # test_labels_IndepAlbedo = all_labels_IndepAlbedo[train_dataset.index]


            # ------------- fit RANDOM FOREST MODEL ----------------------------
            print('Training random forest model')
            scal = StandardScaler()
            # Xrf_train = scal.fit_transform(Xrf_train)
            # Xrf_test = scal.transform(Xrf_test)
            # regressor = RandomForestRegressor(n_estimators=20)
            # regressor.fit(Xrf_train, yrf_train)
            # yrf_pred = regressor.predict(Xrf_test)

            Xrf_test_testalbedo = scal.fit_transform(test_features_testalbedo)
            Xrf_train = scal.fit_transform(train_features)
            Xrf_test = scal.transform(test_features)
            regressor = RandomForestRegressor(n_estimators=100)
            regressor.fit(Xrf_train, train_labels)
            yrf_pred = regressor.predict(Xrf_test)
            yrf_pred_testalbedo = regressor.predict(Xrf_test_testalbedo)

            # get feature importances
            importance = regressor.feature_importances_
            # summarize feature importance
            print('label => {}'.format(label))
            feats = list(train_features.keys())
            print('features => {}'.format(feats))
            for i, v in enumerate(importance):
                print('Feature %s: %0d, Score: %.5f' % (feats[i], i, v))
            # plot feature importance
            # pyplot.bar([x for x in range(len(importance))], importance)
            # pyplot.show()


            # do model predictions
            # yrf_pred_A2 = regressor.predict(scal.transform(Xrf_test))
            # yrf_pred_sameA = regressor.predict(scal.transform(Xrf))
            # ymlr_pred_A2 = mlr_model.predict(Xrf)

            # yrf_pred2 = regressor.predict(Xrf_train)

            # if label == 'fcoup':
            #     yrf_pred.shape
            #     yrf_test.shape
            #     matplotlib.use('Qt5Agg') # dyn show plots
            #     plt.figure()
            #     plt.plot( yrf_pred, yrf_test, 'o')
            #     plt.plot( yrf_pred2, yrf_train, '*r')
            #     plt.plot( yrf_test, yrf_test, 'k')
            #     plt.show()

                # plt.figure()
                # plt.plot()
            #
            #     print('wait!')



            # ------------------------------------------------------------------


            # ------------------ fit NEURAL NETWORK MODEL ----------------------
            # print('Training neural network model')
            # dnn_savedir = os.path.join(datadir, 'dnn_models')
            # dnn_savename = 'dnn_model_{}_cosz_{}_adir_{}'.format(
            #                         label, mycosz,  myadir)
            # dnn_model = dem.train_neuralnet(dfred, label=label,
            #                                 train_frac = train_frac,
            #                                 activation='linear',
            #                                 plot=False,
            #                                 save = True,
            #                                 savedir= dnn_savedir,
            #                                 savename=dnn_savename)
            # ------------------------------------------------------------------


            # --------------- fit MULTIPLE LINEAR REGRESSION -------------------
            mlr_savename = 'mlr_model_{}_cosz_{}_adir_{}'.format(
                                    label, mycosz, test_adir)

            mlr_data = 'mlr_testdata_{}_cosz_{}_adir_{}.csv'.format(
                                    label, mycosz, test_adir)

            # print('Training multiple linear regression')
            # # currently using all values, not just training fraction
            # mlr_model = dem.train_multlr(dfred, label=label,
            #                      train_frac = train_frac,
            #                      plot=False,
            #                      save=True,
            #                      savedir=mlr_savedir,
            #                      savename = mlr_savename,
            #                      testdataname = mlr_data)

            with open(os.path.join(prefit_model_folder,
                        '{}.pickle'.format(mlr_savename)), 'rb') as pklf:
                # to read the saved model, do the following:
                mlr_model = pickle.load(pklf)
                # pass

            ymlr_pred = mlr_model.predict(test_features)
            ymlr_pred_testalbedo = mlr_model.predict(test_features_testalbedo)

            # ------------------------------------------------------------------



            # -------------- compare with WLL parameterization -----------------

            # yrf_A2

            if label not in ['fcoup', 'fcoupn']:
                # currently using all values, not just training fraction
                # ylee_pred_A2 = dem.lee_model_predict(dfvars,
                #            label=label, cosz=mycosz, albedo=test_adir)

                ylee_pred = dem.lee_model_predict(test_dataset,
                            label=label, cosz=mycosz, albedo=test_adir)

                ylee_pred_testalbedo = dem.lee_model_predict(test_dataset_testalbedo,
                                                  label=label, cosz=mycosz, albedo=test_adir)

            # yrf_pred_all = regressor.predict(sc.transform(np.array(dfvars)))
            # ymlr = mlr_model.predict(dfvars)
            # ydnn = dnn_model.predict(dfvars)
            # dfall = pd.concat([dfvars, dfflux], axis=1, join="inner")
            # matplotlib.use('Qt5Agg') # dyn show plots

            # plt.figure(figsize=(10, 10))
            # if label not in ['fcoup', 'fcoupn']:
            #     # ylee = dem.lee_model_predict(dfall,
            #     #                              label=label, cosz=mycosz,
            #     #                              albedo=myadir)
            #     plt.plot(dfflux_testing_albedo[label], ylee_pred_A2, 'or', label='WLL')
            # plt.plot(dfflux_testing_albedo[label], ymlr_pred_A2, '*b', label='MLR')
            # # plt.plot(dfall[label], yrf_pred_all, '^g', label = 'RF')
            # plt.plot(dfflux[label], yrf_pred_sameA, 'sy', label='RF')
            # plt.plot(dfflux_testing_albedo[label], yrf_pred_A2, '^g', label='RF')
            # # plt.plot(dfall[label], ydnn, '*g')
            # plt.plot(yrf_IndepAlbedo, yrf_IndepAlbedo, 'k')
            # plt.title('Lee model - {} flux norm. diff.'.format(label))
            # plt.xlabel('{} - Monte Carlo Simulation'.format(label))
            # plt.ylabel('{} - Lee 2011 parameterization'.format(label))
            # lee_savename = 'lee_model_{}_cosz_{}_adir_{}.png'.format(
            #     label, mycosz, test_adir)
            # plt.savefig(os.path.join(mlr_savedir, lee_savename))
            # # plt.show()
            # plt.close()

            # yrf_pred_all = regressor.predict( sc.transform(np.array(dfvars)))
            # ymlr = mlr_model.predict(dfvars)
            # ydnn = dnn_model.predict(dfvars)
            # dfall = pd.concat([dfvars, dfflux], axis=1, join="inner")
            # matplotlib.use('Qt5Agg') # dyn show plots
            SSres_rf = np.sum((test_labels - yrf_pred)**2)
            SSres_mlr = np.sum((test_labels - ymlr_pred)**2)

            SSres_rf_testalbedo = np.sum((test_labels_testalbedo - yrf_pred_testalbedo)**2)
            SSres_mlr_testalbedo = np.sum((test_labels_testalbedo - ymlr_pred_testalbedo)**2)
            SStot = np.sum((test_labels - np.mean(test_labels))**2)
            SStot_testalbedo = np.sum((test_labels_testalbedo - np.mean(test_labels_testalbedo))**2)
            R2mlr = 1 - SSres_mlr/SStot
            R2rf = 1 - SSres_rf/SStot

            R2mlr_testalbedo = 1 - SSres_mlr_testalbedo/SStot_testalbedo
            R2rf_testalbedo = 1 - SSres_rf_testalbedo/SStot_testalbedo
            plt.figure(figsize=(10, 10))
            if label not in ['fcoup', 'fcoupn']:
                # ylee = dem.lee_model_predict(dfall,
                #         label=label, cosz=mycosz, albedo=myadir)
                plt.plot(test_labels, ylee_pred, 'or', label = 'WLL')
            plt.plot(test_labels, ymlr_pred, 'ob', label = r'MLR $r^2 = {:.2f}$'.format(R2mlr))
            plt.plot(test_labels, yrf_pred, 'og', label = r'RF $R^2 = {:.2f}$'.format(R2rf))

            # plot prediction for independent albedo value
            if label not in ['fcoup', 'fcoupn']:
                plt.plot(test_labels_testalbedo, ylee_pred_testalbedo, '*r', label = 'WLL')
            plt.plot(test_labels_testalbedo, ymlr_pred_testalbedo, '*b', label = r'MLR $r^2 = {:.2f}$'.format(R2mlr_testalbedo))
            plt.plot(test_labels_testalbedo, yrf_pred_testalbedo, '*g', label = r'RF $R^2 = {:.2f}$'.format(R2rf_testalbedo))

            plt.plot(ymlr_pred, ymlr_pred, 'k')
            plt.title('pred {} flux, cosz = {}.'.format(label, mycosz))
            plt.xlabel('{} - Monte Carlo Simulation'.format(label))
            plt.ylabel('{} - parameterization prediction'.format(label))
            # lee_savename = 'lee_model_{}_cosz_{}_adir_{}.png'.format(
            #                         label, mycosz,  myadir)
            lee_savename = 'lee_model_{}_cosz_{}.png'.format(
                label, mycosz)
            plt.legend()
            plt.savefig( os.path.join(mlr_savedir, lee_savename))
            # plt.show()
            plt.close()
            # ------------------------------------------------------------------

            #-------------------------------------------------------------------
            # load models for all scales, and plot them together
            # to check whether parameterization is scale-dependent

            do_scalewise_analysis = False
            if do_scalewise_analysis:
                YMLR_PREDSCALES = []

                for isc in range(naveblocks):

                    # load model for current scale (and cosz)
                    myblock = AVEBLOCKS[isc]

                    mlr_savedir_isc = os.path.join(datadirm,
                            'mlr_models_ave_{}'.format(AVEBLOCKS[isc]))
                    # if os.path.exists(mlr_savedir_isc):
                    with open(os.path.join(mlr_savedir_isc,
                            '{}.pickle'.format(mlr_savename)), 'rb') as pklf:
                        mlr_model_isc = pickle.load(pklf)

                    # reaload data for current averaging block
                    # ds_sin_isc = xr.open_dataset(os.path.join(datadirm,
                    #                                           'train_test_data_size_{}_adir_singles_{}.nc'.format(
                    #                                               AVEBLOCKS[isc], adir_prod)))
                    #
                    # dfvars_isc = pd.DataFrame({
                    #     'ele':  ds_sin_isc['ele'][:].values,
                    #     'sde':  ds_sin_isc['sde'][:].values,
                    #     'sian': ds_sin_isc['sian'][:, ic].values,
                    #     'svfn': ds_sin_isc['svfn'][:].values,
                    #     'tcfn': ds_sin_isc['tcfn'][:].values
                    # })
                    # train_features_isc = dfvars_isc.sample(frac=train_frac, random_state=0)
                    # test_features_isc = train_features_isc.drop(train_features_isc.index)

                    # vary the model but apply to the same test data
                    YMLR_PREDSCALES.append(mlr_model_isc.predict(test_features))



                MARKERS = itertools.cycle((',', '+', '.', '^', '*', 's'))
                COLORS = itertools.cycle(("blue", "red", "green", "orange", "yellow", "cyan"))
                plt.figure(figsize=(10, 10))
                # if label not in ['fcoup', 'fcoupn']:
                    # ylee = dem.lee_model_predict(dfall,
                    #         label=label, cosz=mycosz, albedo=myadir)
                    # plt.plot(test_labels, ylee_pred, 'or', label = 'WLL')
                plt.plot(test_labels, ymlr_pred, 'ob', label = r'MLR $r^2 = {:.2f}$'.format(R2mlr))
                # plt.plot(test_labels, yrf_pred, 'og', label = r'RF $R^2 = {:.2f}$'.format(R2rf))


                for isc in range(naveblocks):


                    plt.plot(test_labels, YMLR_PREDSCALES[isc],
                             marker = next(MARKERS),
                             color=next(COLORS), linestyle='',
                             label = 'scale {}'.format(AVEBLOCKS[isc]))

                # plot prediction for independent albedo value
                # if label not in ['fcoup', 'fcoupn']:
                #     plt.plot(test_labels_testalbedo, ylee_pred_testalbedo, '*r', label = 'WLL')
                # plt.plot(test_labels_testalbedo, ymlr_pred_testalbedo, '*b', label = r'MLR $r^2 = {:.2f}$'.format(R2mlr_testalbedo))
                # plt.plot(test_labels_testalbedo, yrf_pred_testalbedo, '*g', label = r'RF $R^2 = {:.2f}$'.format(R2rf_testalbedo))

                plt.plot(ymlr_pred, ymlr_pred, 'k')
                plt.title('pred {} flux, cosz = {}.'.format(label, mycosz))
                plt.xlabel('{} - Monte Carlo Simulation'.format(label))
                plt.ylabel('{} - parameterization prediction'.format(label))
                # lee_savename = 'lee_model_{}_cosz_{}_adir_{}.png'.format(
                #                         label, mycosz,  myadir)
                lee_savename = 'allscales_model_{}_cosz_{}.png'.format(
                    label, mycosz)
                plt.legend()
                plt.savefig( os.path.join(mlr_savedir, lee_savename))
                # plt.show()
                plt.close()


                # ------------------------------------------------------------------
