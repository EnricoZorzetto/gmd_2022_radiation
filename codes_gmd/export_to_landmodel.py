import os
import numpy as np
import pickle

# read the model coeffs to import them in the model

parent_outdir = os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_output')
prefit_model_buffer = 0.2
prefit_model_aveblock = 110
# prefit_model_buffer = 0.35
prefit_model_adir = 0.3
prefit_model_domain = 'EastAlps'

modeldir = os.path.join( parent_outdir,
                         'res_Peru_vs_EastAlps',
                         # 'res_Peru_vs_EastAlps_buffer01',
                         'trained_models',
                         'domain_EastAlps_buffer_{}'.format(prefit_model_buffer),
                         'models_ave_{}'.format(prefit_model_aveblock),
                         )

COSZ = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
LABELS = ['fdir', 'frdir', 'fdif', 'frdif']


# COSZ = [0.1]
# LABELS = ['fdif']

# mydf = {'elen': self.elen, 'sian': self.sian, 'tcfn': self.tcf, 'svfn': self.svf}
model = 'mlr'
# label = 'fdir'
# cosz = 0.1

do_old_format = False
if do_old_format:
    for il, label in enumerate(LABELS):
        print("! label = {}".format(label))
        for ic, cosz in enumerate(COSZ):
            model_loadname = '{}_model_{}_cosz_{}_adir_{}'.format(
                model, label, cosz, prefit_model_adir)
            with open(os.path.join(modeldir,
                        '{}.pickle'.format(model_loadname)), 'rb') as pklf:
                loaded_model = pickle.load(pklf)
            if ic == 0:
                NAMES = list(loaded_model.feature_names_in_)
                NAMES.append('intercept')
                print( '!_________________________', '________'.join('{}'.format(x) for x in NAMES), '__!')
            COEFFS = list(loaded_model.coef_)
            COEFFS.append(loaded_model.intercept_)
                # print(COEFFS)

            # print( 'cosz = {:.2f} :'.format(cosz), np.array(loaded_model.coef_), np.array( loaded_model.intercept_ ))
            # print('cosz = {:.2f} :'.format(cosz), *np.array(loaded_model.coef_), np.array(loaded_model.intercept_))
            startline = 'data coeff_{}(:,{}) = '.format(label, ic + 1)
            print('{} / '.format(startline), ', '.join('{:.4E}'.format(x) for x in COEFFS), ' /')



for il, label in enumerate(LABELS):
    print("! label = {}".format(label))
    for ic, cosz in enumerate(COSZ):
        model_loadname = '{}_model_{}_cosz_{}_adir_{}'.format(
            model, label, cosz, prefit_model_adir)
        with open(os.path.join(modeldir,
                               '{}.pickle'.format(model_loadname)), 'rb') as pklf:
            loaded_model = pickle.load(pklf)
        if ic == 0:
            NAMES = list(loaded_model.feature_names_in_)
            NAMES.append('intercept')
            print( '!______________________', '________'.join('{}'.format(x) for x in NAMES), '__!')
        COEFFS = list(loaded_model.coef_)
        COEFFS.append(loaded_model.intercept_)
            # print(COEFFS)

        # print( 'cosz = {:.2f} :'.format(cosz), np.array(loaded_model.coef_), np.array( loaded_model.intercept_ ))
        # print('cosz = {:.2f} :'.format(cosz), *np.array(loaded_model.coef_), np.array(loaded_model.intercept_))
        startline = 'coeff_{}(:,{}) = '.format(label, ic + 1)
        print('{} (/ '.format(startline), ', '.join('{:.4E}'.format(x) for x in COEFFS), ' /)')
# ypred = loaded_model.predict(mydf_flattened)
