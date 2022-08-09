#
# import os
# os.system("python main_run_gmd_2021_0.py")
# os.system("python main_run_gmd_2021_G.py")



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

import copy
import main_run_gmd_2021_0 as c0
import main_run_gmd_2021_G as cG


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
    "read_original_output" : True, # to be done only once

    "make_plots" : True, # make goodness of fit plots in the train/test function

    ### FOR CROSS VALIDATON: USE PREVIOUSLY TRAINED MODEL
    ### FOR A GIVEN DOMAIN AND SPATIAL AVERAGING SCALE:
    "test_only" : False, # models already trained
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





gridlist = {
    "domains" : ["EastAlps", "Nepal", "Peru"],
    "npvalues" : [1, 2, 5, 10, 20, 50, 100, 200],
    "NPV_2_PLOT" : [1, 5, 50],
    # "inputkinds" : ['k2n1pV', 'k5n1pV', 'k2nVp2', 'kVn1p2', 'kVn1p5'],
    "inputkinds": ['k5n1pV', 'kVn1p5'],
    "mycosz" : 0.4,
    # myazi = -0.0*np.pi
    "myazi_index" : 0.0,
    # myazi = myazi_index*np.pi
    "myalbedo" : 0.3,

    # by default, compare tiled statistics to those of the original SRTM field (90m)
    # If instead we want to compare it to stats at a different aggregation scale, turn this True
    # and select the size of the averaging block (e.g., hrmap_aveblock = 6 -> size = 90m * 6
    "do_average_hr_fields" : False,
    "hrmap_aveblock" : 32,
    "prefit_model_aveblock" : 6,
    "prefit_avecorrection_aveblock" : 55,
    "prefit_model_buffer" : 0.2,
    "prefit_model_adir" : 0.3,
    "prefit_model_domain" : 'Peru',
    "parent_datadir" : os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_data'),
    "parent_outdir" : os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_output'),
    ########################################################################
    # stats = ['stdv', 'skew', 'kurt']
    "stats" : ['mean', 'stdv', 'skew', 'kurt'],
    "models" : ['MLR'],
    # labels = ['fdir', 'fdif', 'frdirn', 'frdifn','fcoupn']
    # labels = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']
    "labels" : ['fdir', 'fdif', 'frdir', 'frdif'],
    "terrains" : ['sc', 'ss', 'svf', 'tcf', 'elen'],

    "PREDICTORS_TO_USE" : ['elen', 'sian', 'tcfn', 'svfn'],
    "SPECIFIC_PREDS" : {
        'fdir': ['sian', 'svfn'],
        'fdif': ['sian', 'svfn', 'elen'],
        'frdir': ['sian','svfn', 'tcfn'],
        'frdif': ['svfn', 'tcfn'],
        'frdirn': ['sian','svfn', 'tcfn'],
        'frdifn': ['svfn', 'tcfn'],
        'fcoup': ['sian', 'svfn', 'tcfn'],
        'fcoupn': ['sian', 'svfn', 'tcfn']
    },
}

################################################################################


# read data and train models
c0.main_analysis_rmc_results(namelist)


# do cross validation:
namelist2 = copy.deepcopy(namelist)
namelist2['test_only'] = True
c0.main_analysis_rmc_results(namelist2)




# grid analysis
cG.main_grid_analysis(gridlist)
