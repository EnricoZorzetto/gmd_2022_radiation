import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import demfuncs as dem
from topocalc.viewf import viewf, gradient_d8
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error,  mean_absolute_percentage_error
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


############################## ANALYSIS SETTINGS ###############################

################################################################################

namelist = {
# folder with the original RMC model output
# simul_folder = os.path.join('/Users', 'ez6263', 'Documents','dem_datasets')
"simul_folder" : os.path.join('/Users', 'ez6263', 'Documents','gmd_2021', 'gmd_2021_data'),

"result_output_folder" : os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021',
                                    'gmd_2021_output', 'res_Peru_vs_EastAlps'),


# select RMC simulation to process::
"adir_prod" : 'merged123456', # new datasets - ending name convention
# adir_prod = 'run6' # new datasets - ending name convention

# DOMAINS = ['EastAlps'] # domains to use to read data and train models
# DOMAINS = ['Peru']
"DOMAINS" : ['Peru', "EastAlps"],
# DOMAINS = ["EastAlps"]
# CROP_BUFFERS = [0.1, 0.25]
"CROP_BUFFERS" : [0.2],

# AVEBLOCKS =   [12, 55]
"AVEBLOCKS" :   [1, 3, 6, 12, 32, 55, 110],
# AVEBLOCKS =   [55, 110]
# AVEBLOCKS =   [1, 6]
# AVEBLOCKS =   [55]


"AVEBLOCKS_TO_PLOT" :   [1, 6, 55], # must be a subset of AVEBLOCKS
# AVEBLOCKS_TO_PLOT =   [55, 110] # must be a subset of AVEBLOCKS
# for x in AVEBLOCKS_TO_PLOT:
#     assert x in AVEBLOCKS


# LABELS = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn'] # labels to predict
"LABELS" : ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup'], # labels to predict
"do_average" : True,
# crop_buffer = 0.20
"FTOAnorm" : 1361.0, # * mycosz

"resample_data" : True, # extract only of random subset of the training data
"resample_data_size" : 100000,


# do_rf = False
"plot_original_fields" : True, # only done if also read_original_output = True
"read_original_output" : False, # to be done only once

"make_plots" : True, # make goodness of fit plots in the train/test function

### FOR CROSS VALIDATON: USE PREVIOUSLY TRAINED MODEL
### FOR A GIVEN DOMAIN AND SPATIAL AVERAGING SCALE:
"test_only" : True, # models already trained
"test_single_aveblock" : False,
"prefit_models_aveblock" : 6, # use this model for all cases (all aveblocks)
"prefit_models_adir" : 0.3,
# prefit_models_domain = 'Peru'
"prefit_models_domain" : 'Peru',
#################################

# PREDICTORS_TO_USE = ['sian', 'tcfn', 'svfn', 'sde', 'ele']
# PREDICTORS_TO_USE = ['elen', 'sian', 'tcfn', 'svfn']
# PREDICTORS_TO_USE = ['sian', 'tcfn', 'svfn']
# PREDICTORS_TO_USE = ['tcfn', 'svfn','sian']

# MODELS = ['LEE', 'NLR']
"MODELS" : ['MLR', 'RFR', 'LEE', "NLR"],
# MODELS = ['MLR', 'LEE']

"MODELS_TO_PLOT" : ['MLR', 'RFR'], # must be a subset of models
# GOF_METRICS = ['R2', 'MAE', 'MSE', 'EVAR', 'MAXE', 'MAPE']
"GOF_METRICS" : ['R2', 'EVAR'],

# reduce these to single-valued listst [0] unless needed for exploring performance
# set them to lists of length 1 in production runs:
# RF_N_ESTIMATORS = [5, 10, 20, 50, 100]
# RF_MAX_DEPTH = [5, 8, 10, 15, 20]

# optimal parameters::
"RF_N_ESTIMATORS" : [20],
"RF_MAX_DEPTH" : [10],

"do_plot_histograms" : True,

# specific predictors to use for each label
"normalize_Vd_and_Ct" : True,

"PREDICTORS_TO_USE" : ['elen', 'sian', 'tcfn', 'svfn'],

"SPECIFIC_PREDS" : { # specific predictors to use for each label
    'fdir': ['sian', 'svfn'],
    'fdif': ['sian', 'svfn', 'elen'],
    'frdir': ['sian','svfn', 'tcfn'],
    'frdif': ['svfn', 'tcfn'],
    'frdirn': ['sian','svfn', 'tcfn'],
    'frdifn': ['svfn', 'tcfn'],
    'fcoup': ['sian', 'svfn', 'tcfn'],
    'fcoupn': ['sian', 'svfn', 'tcfn']
},

} # end namelist

################################################################################


def main_analysis_rmc_results(nml):
    """
    Read RMC results, and train or test models
    """
    ### FIRST UNPACKS NAMELIST VARIABLES ###
    # --------------------------------------

    # --------------------------------------
    ### END UNPACKING NAMELIST VARIABLES ###

    print('test_only?')
    print(nml['test_only'])

    if nml['test_only']:
        nml['read_original_output'] = False
        print('Testing only: cross validation. data must already have been processed and models trained.')
    else:
        print('Training mode: fitting predictive models + same sample validation')

    # if read_original_output:


    histogram_data = {}



    nlabels = len(nml['LABELS'])
    naveblocks = len(nml['AVEBLOCKS'])
    ndomains = len(nml['DOMAINS'])
    nbuffers = len(nml['CROP_BUFFERS'])
    datadir_diffs = os.path.join(nml['result_output_folder'], 'simul_differences')
    datadir_models = os.path.join(nml['result_output_folder'], 'trained_models')
    datadir_crossval = os.path.join(nml['result_output_folder'], 'crossval_models')
    datadir_gof_metrics = os.path.join(nml['result_output_folder'], 'gof_metrics')
    datadir_gof_histograms = os.path.join(datadir_gof_metrics, 'histograms')

    if not os.path.exists(nml['result_output_folder']):
        os.makedirs(nml['result_output_folder'])
    if not os.path.exists(datadir_diffs):
        os.makedirs(datadir_diffs)
    if not os.path.exists(datadir_models):
        os.makedirs(datadir_models)
    if not os.path.exists(datadir_crossval):
        os.makedirs(datadir_crossval)
    if not os.path.exists(datadir_gof_metrics):
        os.makedirs(datadir_gof_metrics)
    if not os.path.exists(datadir_gof_histograms):
        os.makedirs(datadir_gof_histograms)

    # read_original_output = False # to be done only once

    if nml['read_original_output']:
        print('Reading raw data from RMC simulations ...')
        for id, mydomain in enumerate(nml['DOMAINS']):
            for ibu, mybuffer in enumerate(nml['CROP_BUFFERS']):
                for ibl, myaveblock in enumerate(nml['AVEBLOCKS']):
                    print(ibl)
                    # old lat-lon siumulations
                    # simul_name = 'output_cluster_PP3D_{}_run1_{}'.format(mydomain, adir_prod)

                    # new equal area simulations
                    simul_name = 'output_cluster_PP3D_{}_{}'.format(mydomain, nml['adir_prod'])

                    datadir = os.path.join(nml['simul_folder'], simul_name)
                    # adir_prod = datadir.split('_')[-1] # new convention, name end changes now - same for all

                    outdiffdif = os.path.join(datadir_diffs,
                                'domain_{}_buffer_{}'.format(mydomain, mybuffer))

                    if not os.path.exists(outdiffdif):
                        os.makedirs(outdiffdif)

                    outfigdir_fields = os.path.join(datadir_diffs, 'simul_fields_plots'
                            'plots_{}_buffer_{}_ave_{}'.format(mydomain, mybuffer, myaveblock))
                    if nml['plot_original_fields'] and ibu == 0 and myaveblock in nml['AVEBLOCKS_TO_PLOT']: # plot fields only in this case
                    # if plot_original_fields and ibu == 0 and ibl == 0: # plot fields only in this case
                        do_plots = True
                        os.system("mkdir -p {}".format(outfigdir_fields))
                    else:
                        do_plots = False
                    dem.read_PP_3D_differences(datadir=datadir,
                                               outdir=outdiffdif,
                                               do_average=nml['do_average'],
                                               aveblock=myaveblock,
                                               crop_buffer=mybuffer,
                                               do_plots=do_plots,
                                               outfigdir=outfigdir_fields,
                                               FTOAnorm=nml['FTOAnorm'])




    ################################################################################



    # train models


    print('Training and/or testing models ...')
    for id, mydomain in enumerate(nml['DOMAINS']):
        for ibu, mybuffer in enumerate(nml['CROP_BUFFERS']):
            for ibl, myaveblock in enumerate(nml['AVEBLOCKS']):
                # print(id, ibu, ibl)

                # folder to read the calibration / validation data (is the same in train / test )
                mydatadir_diffs = os.path.join(datadir_diffs,
                            'domain_{}_buffer_{}'.format(mydomain, mybuffer))

                if not nml['test_only']: # we are actually training the models here::
                    test_savedir = None
                    model_savedir = os.path.join(datadir_models,
                            'domain_{}_buffer_{}'.format(mydomain, mybuffer),
                            'models_ave_{}'.format(myaveblock) )
                    if not os.path.exists(model_savedir):
                        os.makedirs(model_savedir)

                else: # testing only - trained model must already exist
                    if not nml['test_single_aveblock']: # do not use same aveblock for model to be tested
                        nml['prefit_models_aveblock'] = myaveblock
                        save_prefit_models_aveblock = 'all'
                    else:
                        save_prefit_models_aveblock = nml['prefit_models_aveblock']

                    # directory with the model to test
                    model_savedir = os.path.join(datadir_models,
                            'domain_{}_buffer_{}'.format(nml['prefit_models_domain'], mybuffer),
                            'models_ave_{}'.format(nml['prefit_models_aveblock']) )
                    # print('testing only:: directory of trained model = {}'.format(model_savedir))
                    if not os.path.exists(model_savedir):
                        raise Exception('Error in model testing: the directory with saved '
                                        'models must already exist!')
                    # create directory with the results / plots for the testing
                    test_savedir = os.path.join(datadir_crossval,
                            'domain_{}_buffer_{}'.format(mydomain, mybuffer),
                            'cv_Train_{}_adir_{}_ave_{}_Test_{}_ave_{}'.format(
                            nml['prefit_models_domain'], nml['prefit_models_adir'], nml['prefit_models_aveblock'],
                            mydomain, myaveblock ))
                    if not os.path.exists(test_savedir):
                        os.makedirs(test_savedir)

                # read albedo-by-albedo results
                ds_sin = xr.open_dataset( os.path.join(mydatadir_diffs,
                            'train_test_data_size_{}_adir_singles_{}.nc'.format(
                            myaveblock, nml['adir_prod'])))

                ADIRs = ds_sin['ADIRs'].values
                # ADIRs = ADIRs[:1]
                # print('WARNING: using a single ADIRs')
                COSZs = ds_sin['COSZs'][:].values
                nadir = np.size(ADIRs)
                ncosz = np.size(COSZs)
                # subset these based on user choice?

                # use the variables not normalized by cosz for perdiction
                if not nml['normalize_Vd_and_Ct']:
                    ds_sin['svfn'] = ds_sin['svf0']
                    ds_sin['tcfn'] = ds_sin['tcf0']

                # initialize xarray dataset to store goodness of fit measures
                if id==0 and ibu==0 and ibl==0:
                    model_metrics = dem.init_metrics_dataset(
                            nml['DOMAINS'], nml['AVEBLOCKS'], nml['CROP_BUFFERS'], ADIRs,
                            nml['LABELS'], COSZs, nml['MODELS'], nml['GOF_METRICS'],
                            nml['RF_N_ESTIMATORS'], nml['RF_MAX_DEPTH'])
                    # modeldata = dem.init_metrics_dataset()

                for ic, mycosz in enumerate(COSZs):
                    for ia, myadir in enumerate(ADIRs):



                        # create dataframe with real (a.k.a., simulated) flux values
                        dfflux = pd.DataFrame({
                            mylab:   ds_sin[mylab][:,   ic, ia].values for mylab in nml['LABELS']})
                        # create dataframe with predictors for testing or training models
                        dfvars = pd.DataFrame({
                            mypred: ds_sin[mypred][:].values for mypred in nml['PREDICTORS_TO_USE']
                            if mypred not in ['sian']})
                        if 'sian' in nml['PREDICTORS_TO_USE']:
                            dfvars['sian'] = ds_sin['sian'][:,ic].values
                        dfvars = dfvars[nml['PREDICTORS_TO_USE']] # re-order columns

                        # dftot = pd.concat([dfflux, dfvars], axis=1).reindex(dfflux.index)
                        # print(dfflux.shape)
                        # print(dfvars.shape)
                        # print(dftot.shape)

                        # resample data to get small sample size for training:
                        # if resample_data and resample_data_size < dfflux.shape[0]:
                            # dftot = pd.concat([dfflux, dfvars])

                        ##### store data later used to produce histogram of current dataset - optional
                        # do_plot_histograms = True
                        if nml['do_plot_histograms']:
                            for key1 in dfflux.columns:
                                histogram_data['{}_{}_{}_{}'.format(mydomain, mycosz, myaveblock, key1)] = dfflux[key1]
                            for key2 in dfvars.columns:
                                histogram_data['{}_{}_{}_{}'.format(mydomain, mycosz, myaveblock, key2)] = dfvars[key2]
                        #####

                        if not nml['test_only']:
                            dftot = pd.concat([dfflux, dfvars], axis=1).reindex(dfflux.index)
                            remove_null_obs = False
                            if remove_null_obs:
                                cond = dftot['fdir'].values > -0.8
                                dftot1 = dftot[cond]
                            else:
                                dftot1 = dftot
                            if nml['resample_data'] and nml['resample_data_size'] < dftot1.shape[0]:
                                dftot2 = dftot1.sample(n = nml['resample_data_size'], random_state=0)
                            else:
                                dftot2 = dftot1
                            dfvars = dftot2[nml['PREDICTORS_TO_USE']]
                            dfflux = dftot2[nml['LABELS']]


                        for ines, rf_n_estim in enumerate(nml['RF_N_ESTIMATORS']):
                            for imtd, rf_max_dep in enumerate(nml['RF_MAX_DEPTH']):

                                # note: models are overwritten, only one model is saved for
                                # each max_depth - num_trees combination
                                # it's fine, need these only for checking model
                                # in prod run only one comb of parameter values is used
                                metr = dem.train_test_models(
                                            dfflux=dfflux, dfvars=dfvars,
                                            LABELS=nml['LABELS'],
                                            MODELS=nml['MODELS'],
                                            GOF_METRICS=nml['GOF_METRICS'],
                                            mycosz=mycosz,
                                            myadir=myadir,
                                            rf_n_estimators=rf_n_estim,
                                            rf_max_depth=rf_max_dep,
                                            test_only=nml['test_only'], make_plot=nml['make_plots'],
                                            prefit_models_adir=nml['prefit_models_adir'],
                                            modeldir=model_savedir,
                                            specific_predictors=nml['SPECIFIC_PREDS'],
                                            testdir=test_savedir)

                                for ilab, mylab in enumerate(nml['LABELS']):
                                    for mymet in nml['GOF_METRICS']:
                                        for mymod in nml['MODELS']:
                                            model_metrics['GOFs'].loc[dict(model=mymod,
                                                    metric=mymet, label=mylab,
                                                    domain=mydomain, buffer=mybuffer,
                                                    iadir=ia, icosz = ic, aveblock=myaveblock,
                                                    rf_n_estimators = rf_n_estim,
                                                    rf_max_depth = rf_max_dep)
                                                    ] = metr["{}_{}".format(mymet, mymod)][ilab]

                        model_metrics.attrs['test_only'] = int(nml['test_only'])
                        if nml['test_only']:
                            model_metrics.attrs['cv_training_domain'] = nml['prefit_models_domain']
                            model_metrics.attrs['cv_training_aveblock'] = save_prefit_models_aveblock

    datadir_gof_data = os.path.join(datadir_gof_metrics, 'data')
    datadir_gof_plots = os.path.join(datadir_gof_metrics, 'plots')
    os.system("mkdir -p {}".format(datadir_gof_data))
    os.system("mkdir -p {}".format(datadir_gof_plots))
    if nml['test_only']: # save metric file tracking domain and aveblock of the model tested
        # if test_single_aveblock:
        #     save_prefit_models_aveblock = prefit_models_aveblock
        # else:
        #     save_prefit_models_aveblock = 'all'

        file_res = os.path.join(datadir_gof_data,
                'metric_test_trained_{}_adir_{}_ave_{}.nc'.format(
                nml['prefit_models_domain'], nml['prefit_models_adir'], save_prefit_models_aveblock))
    else:
        file_res = os.path.join(datadir_gof_data,
                'metric_training_models.nc')
    model_metrics.to_netcdf(file_res)

    ### PLOT RESULTS:::
    # outfigdir_gof = datadir_models
    print("plotting goodness of fit measures...")
    dem.plot_gof_measures(file_res, datadir_gof_plots,  METRICS_TO_PLOT = nml['GOF_METRICS'], MODELS_TO_PLOT=nml['MODELS_TO_PLOT'])
    # plot_gof_measures(file_res, datadir_gof_plots,  METRICS_TO_PLOT = GOF_METRICS, MODELS_TO_PLOT=MODELS_TO_PLOT)

    # plot_gof_measures(file_res, datadir_gof_plots,  METRICS_TO_PLOT = GOF_METRICS)
    # plot_gof_measures(file_res, datadir_gof_plots, METRICS_TO_PLOT = GOF_METRICS)
    # plot_gof_measures(file_res, datadir_gof_plots)
    # plot_gof_measures(file_res, datadir_gof_plots)

    if len(nml['RF_N_ESTIMATORS']) > 1: # explore RF performance varying hyper param
        dem.plot_rf_hyper_params(file_res, datadir_gof_plots)


    # now read data




    ### plot histograms and save data

    outfile_hist_data = os.path.join(datadir_gof_histograms, 'histograms_full_data.pkl')
    with open(outfile_hist_data, "wb") as ff1:
        pickle.dump(histogram_data, ff1)

    # pickle_file will be closed at this point, preventing your from accessing it any further

    with open(outfile_hist_data, "rb") as ff2:
        histogram_data2 = pickle.load(ff2)

    ### plot histograms at last

    COSZs_TO_HIST = [0.1, 0.25, 0.55, 0.7]
    AVEBLOCKS_TO_HIST =   [1, 6, 32, 55, 110] # must be a subset of AVEBLOCKS
    AVELABELS_TO_HIST = [0.1, 0.5, 3, 5, 10] # equiv scale in km
    # AVEBLOCKS_TO_PLOT =   [55, 110] # must be a subset of AVEBLOCKS
    for x in AVEBLOCKS_TO_HIST:
        assert x in nml['AVEBLOCKS']
    for x in COSZs_TO_HIST:
        assert x in COSZs

    # AVELABELS_TO_HIST = [ round(x*90/1000) for x in AVEBLOCKS_TO_HIST]
    # DOMAINS
    # AVEBLOCKS_TO_PLOT

    # round(5.7)

    nh_cosz = len(COSZs_TO_HIST); nh_aveb = len(AVEBLOCKS_TO_HIST)
    figsizex = 4*nh_aveb
    figsizey = 4*nh_cosz
    # key2plot = 'fdir';
    domain0 = 'EastAlps'; domain1 = 'Peru' # for now fix a flux term -> one plot for each
    # histogram_data['{}_{}_{}_{}'.format(mydomain, mycosz, myaveblock, key1)] = dfflux[key1]
    KEY2PLOT = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']
    KEY2LABEL = [r'$f_{dir}$', r'$f_{dif}$', r'$f_{rdir}$', r'$f_{rdif}$', r'$f_{coup}$']

    print('producing histogram plots .... ')
    for key2plot, key2label in zip (KEY2PLOT, KEY2LABEL):

        fig, axes = plt.subplots(ncols=nh_aveb, nrows=nh_cosz, figsize=(figsizex, figsizey))
        for iy, myh_cosz in enumerate(COSZs_TO_HIST):
            for ix, myh_aveb in enumerate(AVEBLOCKS_TO_HIST):
                # countxy = ix*nh_cosz + iy
                # print(countxy)
                datah0 = histogram_data2['{}_{}_{}_{}'.format(domain0, myh_cosz, myh_aveb, key2plot)]
                datah1 = histogram_data2['{}_{}_{}_{}'.format(domain1, myh_cosz, myh_aveb, key2plot)]
                nobsh = np.size(datah0)
                # print(datah0.shape)
                # print(datah1.shape)
                # nbinsh = max(int(nobsh//30), 15)
                # axes[iy,ix].hist(datah0, bins=nbinsh, density=True, alpha=0.9, label=domain0)
                # axes[iy,ix].hist(datah1, bins=nbinsh, density=True, alpha=0.5, label=domain1)

                axes[iy,ix].hist(datah0, density=True, alpha=0.9, label=domain0)
                axes[iy,ix].hist(datah1, density=True, alpha=0.5, label=domain1)
                if iy==0:
                    # axes[iy,ix].set_title('scale={}'.format(myh_aveb))
                    axes[iy, ix].set_title('scale={} km'.format(AVELABELS_TO_HIST[ix]))
                if ix == 0:
                    axes[iy, ix].set_ylabel('cosz={}'.format(myh_cosz))
                if iy==0 and ix==0:
                    axes[iy,ix].legend()
                if iy==nh_cosz-1:
                    axes[iy,ix].set_xlabel(key2label)
        plt.tight_layout()
        plt.savefig(os.path.join(datadir_gof_histograms, 'histograms_{}.png'.format(key2plot)))
        # plt.show()
        plt.close()


    # for key2plot, key2label in zip (KEY2PLOT, KEY2LABEL):
    xlabv = 'sian'
    ylabv = 'fdir'
    print('producing scatter plots .... ')
    fig, axes = plt.subplots(ncols=nh_aveb, nrows=nh_cosz, figsize=(figsizex, figsizey))
    for iy, myh_cosz in enumerate(COSZs_TO_HIST):
        for ix, myh_aveb in enumerate(AVEBLOCKS_TO_HIST):
            # countxy = ix*nh_cosz + iy
            # print(countxy)
            datah0x = histogram_data2['{}_{}_{}_{}'.format(domain0, myh_cosz, myh_aveb, xlabv)]
            datah1x = histogram_data2['{}_{}_{}_{}'.format(domain1, myh_cosz, myh_aveb, xlabv)]

            datah0y = histogram_data2['{}_{}_{}_{}'.format(domain0, myh_cosz, myh_aveb, ylabv)]
            datah1y = histogram_data2['{}_{}_{}_{}'.format(domain1, myh_cosz, myh_aveb, ylabv)]

            # nobsh = np.size(datah0)
            # print(datah0.shape)
            # print(datah1.shape)
            # nbinsh = max(int(nobsh//30), 15)
            # axes[iy,ix].hist(datah0, bins=nbinsh, density=True, alpha=0.9, label=domain0)
            # axes[iy,ix].hist(datah1, bins=nbinsh, density=True, alpha=0.5, label=domain1)

            axes[iy,ix].scatter(datah0x, datah0y, marker='o', c="darkred", label=domain0)
            axes[iy,ix].scatter(datah1x, datah1y, marker='*', c="darkblue", label=domain1, alpha=0.3)
            if iy==0:
                # axes[iy,ix].set_title('scale={}'.format(myh_aveb))
                axes[iy, ix].set_title('scale={} km'.format(AVELABELS_TO_HIST[ix]))
            if ix == 0:
                axes[iy, ix].set_ylabel('cosz={} \n {}'.format(myh_cosz, ylabv))
            if iy==0 and ix==0:
                axes[iy,ix].legend()
            if iy==nh_cosz-1:
                axes[iy,ix].set_xlabel(xlabv)
    plt.tight_layout()
    plt.savefig(os.path.join(datadir_gof_histograms, 'scatter_{}_vs_{}.png'.format(xlabv, ylabv)))
    # plt.show()
    plt.close()

    print('done!.')


if __name__ == '__main__':
    main_analysis_rmc_results(namelist)


