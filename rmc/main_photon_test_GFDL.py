
# test the photon tracinf Algorithm over a domain in the Alps


import os
import sys
import json
import time
import numpy as np
import pandas as pd
import photon_mc_land as land
import photon_mc_atmosphere as atm
import photon_mc_numba_fractional as ph
import xarray as xr
import pandas as pd
import Py6S
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# import matplotlib
# import photon_mc_numba_fractional as ph
import matplotlib.ticker
import pickle




ph.matplotlib_update_settings()


from numba import types
from numba.typed import Dict

from netCDF4 import Dataset
# run_from_cluster     = True

# if run_from_cluster:



################################################################################


################################################################################

# Wavelength dependent values must be input at the following wavelengths
# (given in micrometers):
# LAMBDAS = np.array([0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515,
#                     0.550, 0.590, 0.633, 0.670, 0.694, 0.760, 0.860,
#                     1.240, 1.536, 1.650, 1.950, 2.250, 3.750])
# nlambdas = len(LAMBDAS)
# ################################################################################


datadir = "/Users/ez6263/Documents/rmc_datasets/"
# datadir = os.path.join('//', 'home', 'enrico', 'Documents', 'dem_datasets')
# resdir = os.path.join(datadir, 'output_cluster_Py6S_noAerosol')
# resdir = os.path.join(datadir, 'output_multiband')
resdir = os.path.join(datadir, 'output_GFDL_test_ENFRAC')
netcdfdir_3D = os.path.join(resdir, 'output_sim', 'output_sim_3D')
netcdfdir_PP = os.path.join(resdir, 'output_sim', 'output_sim_PP')

# mdfile = os.path.join('exp', 'cluster_test_py6S.json')
mdfile = os.path.join(resdir, 'experiment.json')
# mdfile = os.path.join('exp', 'laptop_test_py6S.json')
metadata = json.load(open(mdfile, 'r'))

dfres_PP = pd.read_csv( os.path.join(resdir, 'list_sim_cases_PP.csv'))
dfres_3D = pd.read_csv( os.path.join(resdir, 'list_sim_cases_3D.csv'))

# with open(os.path.join(resdir, 'dfatm.pickle'), 'r') as handle:
#     b = pickle.load(handle)



outfigdir = os.path.join(resdir, 'outfigdir')
# outdir = os.path.join(resdir, 'output_{}')
# if not os.path.exists(outdir):
#     os.makedirs(outdir)
if not os.path.exists(outfigdir):
    os.makedirs(outfigdir)

aerosol = metadata['aerosol']
# datadir = metadata['datadir']
# exp_name = metadata['exp_name']
MYADIR = metadata['ADIR']
MYCOSZ = metadata['COSZ']
MYFREQ = metadata['BANDS']

# MYFREQ = MYFREQ[3:4].copy()
# MYCOSZ = MYCOSZ[0:1].copy()


count = 0

nfreq = len(MYFREQ)
ncosz = len(MYCOSZ)



VARS = ['EDIR', 'EDIF', 'ECOUP', 'ERDIR', 'ERDIF', 'eTOA', 'eABS', 'eSRF',
        'cosz', 'freq', 'planepar', 'FTOAdown']
# RES_PP = {key:np.zeros(nfreq) for key in VARS}
nrows = 2*nfreq*ncosz
RES = pd.DataFrame({key:np.zeros(nrows) for key in VARS})

for icz in range(ncosz):
    for ifr in range(nfreq):


        # start with these values
        # myalbedo = 0.3
        # mycosz = 0.4

        # myaltitude = 0.0 # [km] target elevation (surface)
        # myaltitude = 1.3 # [km] target elevation (surface) - must match Z average
        # variable parameters
        myalbedo = MYADIR[0]
        mycosz = MYCOSZ[icz] # FIRST VAL
        myfreq = MYFREQ[ifr]
        # mylambda = LAMBDAS[myfreq]


        ################################  READ RMC RESULTS #############################

        unique_phis = np.unique(dfres_3D['phi'])
        myphi = unique_phis[0]
        myrun_PP = dfres_PP.index[(dfres_PP['cosz']==mycosz)     \
                                  & (dfres_PP['myfreq'] == myfreq)]
        myrun_3D = dfres_3D.index[(dfres_3D['cosz']==mycosz)    \
                                & (dfres_3D['myfreq'] == myfreq)  \
                                & (dfres_3D['phi'] == myphi)]

        if len(myrun_PP)==1:
            myrun_PP = myrun_PP[0]
        else:
            raise Exception("Error - PP - too many values for this case!")

        if len(myrun_3D)==1:
            myrun_3D = myrun_3D[0]
        else:
            raise Exception("Error - 3D - too many values for this case!")

        file_ici_PP = os.path.join(netcdfdir_PP,
                        'photonmc_output_{}.nc'.format(myrun_PP))
        file_ici_3D = os.path.join(netcdfdir_3D,
                                   'photonmc_output_{}.nc'.format(myrun_3D))
        dsPP = xr.open_dataset(file_ici_PP)
        ds3D = xr.open_dataset(file_ici_3D)

        # SUM GPOINTS CUMULATIVELY
        RES_PP = {}
        RES_3D = {}

        RES_PP['EDIR']  =  np.sum(dsPP['edir'][:].values) # fluxes in W m^-2 mum^-1 already
        RES_PP['EDIF']  =  np.sum(dsPP['edif'][:].values)
        RES_PP['ECOUP'] = np.sum(dsPP['ecoup'][:].values)
        RES_PP['ERDIR'] = np.sum(dsPP['erdir'][:].values)
        RES_PP['ERDIF'] = np.sum(dsPP['erdif'][:].values)
        RES_PP['eTOA']  = dsPP.attrs['etoa'] # fluxes in W m^-2 mum^-1 already
        RES_PP['eABS']  = dsPP.attrs['eabs']
        RES_PP['eSRF']  = dsPP.attrs['esrf']


        RES_3D['EDIR']  =  np.sum(ds3D['edir'][:].values) # fluxes in W m^-2 mum^-1 already
        RES_3D['EDIF']  =  np.sum(ds3D['edif'][:].values)
        RES_3D['ECOUP'] = np.sum(ds3D['ecoup'][:].values)
        RES_3D['ERDIR'] = np.sum(ds3D['erdir'][:].values)
        RES_3D['ERDIF'] = np.sum(ds3D['erdif'][:].values)
        RES_3D['eTOA']  = ds3D.attrs['etoa'] # fluxes in W m^-2 mum^-1 already
        RES_3D['eABS']  = ds3D.attrs['eabs']
        RES_3D['eSRF']  = ds3D.attrs['esrf']

        ETOAPP = dsPP.attrs['ftoanorm']
        ETOA3D = ds3D.attrs['ftoanorm']

        RES.iloc[count, RES.columns.get_loc('EDIR')] = RES_PP['EDIR']
        RES.iloc[count, RES.columns.get_loc('EDIF')] = RES_PP['EDIF']
        RES.iloc[count, RES.columns.get_loc('ECOUP')] = RES_PP['ECOUP']
        RES.iloc[count, RES.columns.get_loc('ERDIR')] = RES_PP['ERDIR']
        RES.iloc[count, RES.columns.get_loc('ERDIF')] = RES_PP['ERDIF']
        RES.iloc[count, RES.columns.get_loc('eTOA')] = RES_PP['eTOA']
        RES.iloc[count, RES.columns.get_loc('eABS')] = RES_PP['eABS']
        RES.iloc[count, RES.columns.get_loc('eSRF')] = RES_PP['eSRF']
        RES.iloc[count, RES.columns.get_loc('FTOAdown')] = ETOAPP

        RES.iloc[ count + 1, RES.columns.get_loc('EDIR')] = RES_3D['EDIR']
        RES.iloc[ count + 1, RES.columns.get_loc('EDIF')] = RES_3D['EDIF']
        RES.iloc[ count + 1, RES.columns.get_loc('ECOUP')] = RES_3D['ECOUP']
        RES.iloc[ count + 1, RES.columns.get_loc('ERDIR')] = RES_3D['ERDIR']
        RES.iloc[ count + 1, RES.columns.get_loc('ERDIF')] = RES_3D['ERDIF']
        RES.iloc[ count + 1, RES.columns.get_loc('eTOA')] = RES_3D['eTOA']
        RES.iloc[ count + 1, RES.columns.get_loc('eABS')] = RES_3D['eABS']
        RES.iloc[ count + 1, RES.columns.get_loc('eSRF')] = RES_3D['eSRF']
        RES.iloc[ count + 1, RES.columns.get_loc('FTOAdown')]= ETOA3D





        RES.iloc[count, RES.columns.get_loc('planepar')] = 'PP'
        RES.iloc[count + 1, RES.columns.get_loc('planepar')] = '3D'
        RES.iloc[count, RES.columns.get_loc('cosz')] = mycosz
        RES.iloc[count + 1, RES.columns.get_loc('cosz')] = mycosz
        RES.iloc[count, RES.columns.get_loc('freq')] = myfreq
        RES.iloc[count + 1, RES.columns.get_loc('freq')] = myfreq
        count +=2

# RESCUM = np.cumsum(RES)
RESPP = RES[(RES['planepar']=='PP') & (RES['cosz']==mycosz)].copy()
RES3D = RES[(RES['planepar']=='3D') & (RES['cosz']==mycosz)].copy()

plt.figure()
# plt.plot(RESPP['freq'], RESPP['FTOAdown'], '-o')
# plt.plot(RESPP['freq'], RES3D['FTOAdown'], '-o')
plt.plot(RESPP['freq'], RESPP['EDIF'], '-o')
plt.plot(RESPP['freq'], RES3D['EDIF'], '-o')
plt.show()

RESPP_CUM = RESPP.cumsum()
RES3D_CUM = RES3D.cumsum()

dDIR =  (RES3D_CUM['EDIR'].values - RESPP_CUM['EDIR'].values)/RESPP_CUM['EDIR'].values
dDIF =  (RES3D_CUM['EDIF'].values - RESPP_CUM['EDIF'].values)/RESPP_CUM['EDIF'].values
dCOUP = (RES3D_CUM['ECOUP'].values - RESPP_CUM['ECOUP'].values)/RESPP_CUM['ECOUP'].values
dRDIR = (RES3D_CUM['ERDIR'].values)/RESPP_CUM['EDIR'].values
dRDIF = (RES3D_CUM['ERDIF'].values)/RESPP_CUM['EDIF'].values


dimDIR =  (RES3D_CUM['EDIR'].values - RESPP_CUM['EDIR'].values)
dimDIF =  (RES3D_CUM['EDIF'].values - RESPP_CUM['EDIF'].values)
dimCOUP = (RES3D_CUM['ECOUP'].values - RESPP_CUM['ECOUP'].values)
dimRDIR = (RES3D_CUM['ERDIR'].values)
dimRDIF = (RES3D_CUM['ERDIF'].values)

# RESDIFF_CUM = (RES3D_CUM - RESPP_CUM)/RESPP_CUM

plt.figure(figsize = (9, 7))
plt.plot(MYFREQ, dDIR,  '-o',label = 'direct (3D-PP)/PP')
plt.plot(MYFREQ, dDIF,  '-o',label = 'diffuse (3D-PP)/PP')
plt.plot(MYFREQ, dCOUP, '-o', label = 'coupled (3D-PP)/PP')
plt.plot(MYFREQ, dRDIR, '-o', label = 'reflected-direct 3D/PP')
plt.plot(MYFREQ, dRDIF, '-o', label = 'reflected-diffuse 3D/PP')
plt.xlabel('cumulative number of g-points')
plt.ylabel('Normalized flux difference')
plt.title(r'Band 16000 - 22650 $\mathrm{cm}^{-1}$')
plt.legend()
plt.savefig( os.path.join(outfigdir, 'norm_diff.png'), dpi = 300)
plt.show()


plt.figure(figsize = (9, 7))
plt.plot(MYFREQ, dimDIR,  '-o',label = 'direct 3D-PP')
plt.plot(MYFREQ, dimDIF,  '-o',label = 'diffuse 3D-PP')
plt.plot(MYFREQ, dimCOUP, '-o', label = 'coupled 3D-PP')
plt.plot(MYFREQ, dimRDIR, '-o', label = 'reflected-direct 3D')
plt.plot(MYFREQ, dimRDIF, '-o', label = 'reflected-diffuse 3D')
plt.xlabel('cumulative number of g-points')
plt.ylabel(r'Flux [$\mathrm{W \; m}^{-2}$]')
plt.title(r'Band 16000 - 22650 $\mathrm{cm}^{-1}$')
plt.legend()
plt.savefig( os.path.join(outfigdir, 'abs_diff.png'), dpi = 300)
plt.show()
