
# test the photon tracinf Algorithm over a domain in the Alps


import os
import sys
import json
import time
import numpy as np
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
LAMBDAS = np.array([0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515,
                    0.550, 0.590, 0.633, 0.670, 0.694, 0.760, 0.860,
                    1.240, 1.536, 1.650, 1.950, 2.250, 3.750])
nlambdas = len(LAMBDAS)
# ################################################################################


go_fast = False
# datadir = os.path.join('//', 'home', 'enrico', 'Documents', 'dem_datasets')
# datadir = os.path.join('//', 'home', 'enrico', 'Documents', 'dem_datasets')
# datadir = "/home/ez23/dem_datasets/"
datadir = "/Users/ez6263/Documents/rmc_datasets/"
# resdir = os.path.join(datadir, 'output_cluster_Py6S')
resdir = os.path.join(datadir, 'output_Py6S_aerosol_ENFRAC')
# resdir = os.path.join(datadir, 'output_cluster_Py6S_aerosol')
# resdir = os.path.join(datadir, 'output_Py6S')
os.listdir(resdir)
netcdfdir = os.path.join(resdir, 'output_sim', 'output_sim_PP')

# mdfile = os.path.join('exp', 'cluster_test_py6S.json')
mdfile = os.path.join(resdir, 'experiment.json')
# mdfile = os.path.join('exp', 'laptop_test_py6S.json')
metadata = json.load(open(mdfile, 'r'))

dfres = pd.read_csv( os.path.join(resdir, 'list_sim_cases_PP.csv'))

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
MYCOSZ = MYCOSZ[0:1].copy() # only one cosz value

# ncases =

# dfresults = pd.DataFrame()


# resMC['frac_abs_dir'] = EDIR / (ETOA_MC)
# resMC['frac_abs_dif'] = EDIF / (ETOA_MC)
# resMC['frac_abs_cou'] = ECOUP / (ETOA_MC)
# # resMC['frac_inc_dir']  = EDIR / ETOA_6S  / (1-myalbedo)
# # resMC['frac_inc_dif']  = EDIF / ETOA_6S  / (1-myalbedo)
# # resMC['frac_inc_cou'] = ECOUP / ETOA_6S / (1-myalbedo)
# resMC['frac_toa_upw'] = eTOA / ETOA_MC
# resMC['frac_abs_atm'] = eABS / ETOA_MC
# resMC['frac_abs_srf'] = resMC['frac_abs_dir'] + \
#                         resMC['frac_abs_dif'] + resMC['frac_abs_cou']

cosz2plot = 0.5 # MAKE SURE THIS VALUE IS INCLUDED IN THE SIMULATION RESULTS
# RES3D = {}
# RESPP = {}

# start_count = 0

nfreq = len(MYFREQ)
ncosz = len(MYCOSZ)

COLNAMES = ['model', 'cosz', 'freq', 'lambda', 'frac_abs_dir', 'frac_abs_dif', 'frac_abs_cou',
            'frac_toa_upw', 'frac_abs_atm', 'frac_abs_srf', 'FTOAdown']
ncols = len(COLNAMES)
nrows = 2*ncosz*nfreq
datamat = np.zeros((nrows, ncols))
# RES = pd.DataFrame({key:np.ones(nrows) for key in COLNAMES})
RES = pd.DataFrame(datamat, columns=COLNAMES)
# RES = RES.copy()

count = 0

for icz in range(ncosz):
    for ifr in range(nfreq):

        print("*******************************")
        print('count = ', count)
        print("*******************************")

        # start with these values
        # myalbedo = 0.3
        # mycosz = 0.4

        # myaltitude = 0.0 # [km] target elevation (surface)
        # myaltitude = 1.3 # [km] target elevation (surface) - must match Z average
        # variable parameters
        myalbedo = MYADIR[0]
        mycosz = MYCOSZ[icz] # FIRST VAL
        myfreq = MYFREQ[ifr]
        mylambda = LAMBDAS[myfreq]


        ################################  READ RMC RESULTS #############################

        myrun = dfres.index[(dfres['cosz']==mycosz) & (dfres['myfreq'] == myfreq)]

        if len(myrun)==1:
            myrun = myrun[0]
        else:
            raise Exception("Error - too many values for this case!")

        file_ici = os.path.join(netcdfdir,
                        'photonmc_output_{}.nc'.format(myrun))

        ds = xr.open_dataset(file_ici)

        myaltitude = ds.attrs['ave_elev']/1000.0 # elevation in km for py6S

        EDIR = np.sum(ds['edir'][:].values) # fluxes in W m^-2 mum^-1 already
        EDIF = np.sum(ds['edif'][:].values)
        ECOUP = np.sum(ds['ecoup'][:].values)
        ERDIR = np.sum(ds['erdir'][:].values)
        ERDIF = np.sum(ds['erdif'][:].values)
        print(EDIR, EDIF, ECOUP, ERDIR, ERDIF)

        eTOA = ds.attrs['etoa'] # fluxes in W m^-2 mum^-1 already
        eABS = ds.attrs['eabs']
        eSRF = ds.attrs['esrf']



        resMC = {}

        ETOA_MC = ds.attrs['ftoanorm']

        resMC['frac_abs_dir']  = EDIR / (ETOA_MC)
        resMC['frac_abs_dif']  = EDIF / (ETOA_MC)
        resMC['frac_abs_cou'] = ECOUP / (ETOA_MC)
        # resMC['frac_inc_dir']  = EDIR / ETOA_6S  / (1-myalbedo)
        # resMC['frac_inc_dif']  = EDIF / ETOA_6S  / (1-myalbedo)
        # resMC['frac_inc_cou'] = ECOUP / ETOA_6S / (1-myalbedo)
        resMC['frac_toa_upw'] =  eTOA / ETOA_MC
        resMC['frac_abs_atm'] =  eABS / ETOA_MC
        resMC['frac_abs_srf'] = resMC['frac_abs_dir'] + \
                                 resMC['frac_abs_dif'] + resMC['frac_abs_cou']






        # print('fraction of photons incident on surface, dir, dif, coup::')
        # print( 'MC frac inc dir', np.sum(EDIR) / nphotons /  mycosz / (1-alpha_dir) )
        # print( 'MC frac inc dif', np.sum(EDIF) / nphotons /  mycosz / (1-alpha_dir) )
        # print( 'MC frac inc coup', np.sum(ECOUP)/ nphotons / mycosz / (1-alpha_dir) )
        #
        # print('fraction of abs photons dir, dif, coup')
        # print( 'MC frac abs dir', np.sum(EDIR)  /  mycosz / nphotons )
        # print( 'MC frac abs dif', np.sum(EDIF)  /  mycosz / nphotons )
        # print( 'MC frac abs coup', np.sum(ECOUP) / mycosz / nphotons )
        ################################################################################



        ################################################################################
        # Make sure the settings used here are the same used in the atmosphere model

        sat_level = 99.9 # km # here TOA to compute TOA reflectivit
        # sat_level = 60 # km # here TOA to compute TOA reflectivit
        s = Py6S.SixS()
        # ETOAup = atm.get_toa_up_irradiance_py6S(s, nazi=8, nzen=5)
        if aerosol:
            # s.aero_profile = Py6S.AeroProfile.JungePowerLawDistribution(0.1, 0.3)
            s.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Maritime)
            # s.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Urban)
        else:
            s.aero_profile = Py6S.AeroProfile.NoAerosols
        # s.atmos_profile = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

        s.atmos_profile = Py6S.AtmosProfile.PredefinedType(
            Py6S.AtmosProfile.MidlatitudeSummer)




        s.wavelength = Py6S.Wavelength(mylambda)



        s.altitudes.set_target_custom_altitude(myaltitude)


        # s.altitudes.set_sensor_custom_altitude(sat_level)
        s.altitudes.set_sensor_custom_altitude(98)


        s.ground_reflectance = Py6S.GroundReflectance.HomogeneousLambertian(myalbedo)
        s.geometry.solar_z = np.arccos(mycosz)*180.0/np.pi
        s.geometry.view_z = 0.0 # satellite on TOP to compute upward irradiance
        s.run()
        # s.write_input_file(filename = 'input.txt')
        print(s.outputs.fulltext)


        ################################################################################

        # extract variables of interest from the 6S run:
        res6S = {}

        vals = s.outputs.values
        for key in list(vals.keys()):
            print(key, vals[key])

        EDIR_6S = s.outputs.direct_solar_irradiance
        EDIF_6S = s.outputs.diffuse_solar_irradiance
        ECOUP_6S = s.outputs.environmental_irradiance
        ETOA_6S_spectrum = s.outputs.solar_spectrum
        ETOA_6S = ETOA_6S_spectrum * mycosz

        res6S['frac_inc_dir'] =  EDIR_6S  /  ETOA_6S # norm by TOA flux
        res6S['frac_inc_dif'] =  EDIF_6S  /  ETOA_6S
        res6S['frac_inc_cou'] =  ECOUP_6S /  ETOA_6S
        res6S['frac_abs_dir'] =  res6S['frac_inc_dir'] * (1-myalbedo)
        res6S['frac_abs_dif'] =  res6S['frac_inc_dif'] * (1-myalbedo)
        res6S['frac_abs_cou'] =  res6S['frac_inc_cou'] * (1-myalbedo)

        res6S['frac_inc_srf'] = (res6S['frac_inc_dir'] +
                                 res6S['frac_inc_dif'] + res6S['frac_inc_cou'])
        res6S['frac_abs_srf'] = res6S['frac_inc_srf'] * (1-myalbedo)

        res6S['frac_noninc_tot'] = 1 - res6S['frac_inc_srf']

        # res6S['frac_inc_srf']*87.53

        radatm = s.outputs.atmospheric_intrinsic_radiance
        radback = s.outputs.background_radiance
        radapp = s.outputs.apparent_radiance*np.pi
        radpix = s.outputs.pixel_radiance
        radtot = radatm + radback + radpix
        # print(radatm, radback, radpix)
        # print(radtot)
        # print(radapp)

        # res6S['frac_toa_up'] = s.outputs.pixel_radiance*np.pi
        # toa1 = s.outputs.pixel_reflectance
        # toa2 = s.outputs.atmospheric_intrinsic_reflectance
        # toa3 = s.outputs.background_reflectance
        # toa4 = s.outputs.apparent_reflectance*np.pi
        # res6S['frac_abs_surf'] = (EDIR_6S + EDIF_6S + ECOUP_6S)/ETOA_6S*(1-myalbedo) / mycosz
        # res6S['frac_abs_atm'] = 1 - res6S['frac_abs_surf'] - res6S['frac_toa_up']
        atmabs = (1 - s.outputs.total_gaseous_transmittance)

        if go_fast:
            ETOAup = 0
            print('WARNING: We are not computing ETOAup')
        else:
            # ETOAup = atm.get_toa_up_irradiance_py6S(s, nazi = 16, nzen=20)
            ETOAup = atm.get_toa_up_irradiance_py6S(s, nazi=8, nzen=5)
        # ETOAup = get_toa_up_irradiance_py6S(s)
        #
        res6S['flux_toa_upw'] = ETOAup
        res6S['frac_toa_upw'] = res6S['flux_toa_upw'] / ETOA_6S
        res6S['frac_abs_atm'] = 1 -  res6S['frac_toa_upw'] - res6S['frac_abs_srf']
        ################################################################################
        ###################################   END 6S  ##################################

        # df3D = pd.DataFrame(res6S, index = [0])

        # WL = np.linspace(0.3, 0.7, 20)
        # nwl = len(WL)
        # DF = np.zeros(nwl)
        # DD = np.zeros(nwl)
        # DC = np.zeros(nwl)
        # DT = np.zeros(nwl)
        # for iw in range(nwl):
        #     print(iw)
        #     s.run()
        #     s.wavelength = Py6S.Wavelength(WL[iw])
        #     DF[iw] = s.outputs.diffuse_solar_irradiance
        #     DD[iw] = s.outputs.direct_solar_irradiance
        #     DC[iw] = s.outputs.environmental_irradiance
        #     DT[iw] = s.outputs.solar_spectrum
        #
        # plt.figure()
        # plt.plot(WL, DF, '-o', label='dif')
        # plt.plot(WL, DD, '-o', label='dir')
        # plt.plot(WL, DC, '-o', label='coup')
        # plt.plot(WL, DT, '-o', label='TOA')
        # plt.legend()
        # plt.show()

        # DC[DC<0]
        # DF[DF<0]

        # compute percent differences

        orig_stdout = sys.stdout
        outfilename = os.path.join(datadir, 'out_6S_cosz_{}.txt'.format(mycosz))
        # if os.path.exists(outfilename):
        #     print('outfile already there')
        # f = open(outfilename, 'a')
        # sys.stdout = f

        print('-----------------------------------------------------------------------')
        print('lambda = {} [mum]; cosz = {}'.format(mylambda, np.abs(mycosz)))
        print('Solar spectrum value = {} [W m^-2 mum^-1]'.format(ETOA_6S/mycosz))
        print('Normal flux TOA E*cosz = {} [W m^-2 mum^-1]'.format( np.abs(ETOA_6S)))
        allkeys = resMC.keys()
        percdiff = {}
        print('')
        print('NORMALIZED DIFFERENCES   100*(MC - 6S)/6S ')
        for key in allkeys:
            if res6S[key] > 1E-6:
                percdiff[key] = 100.0*(resMC[key] - res6S[key])/res6S[key]
            else:
                percdiff[key] = np.nan
            print('norm diff  {}   {:.3f}'.format(key, percdiff[key]))

        print('')
        print('DIM.LESS [-]    6S    MC')
        for key in allkeys:
            print('{}  {:.3f}   {:.3f}'.format(key, res6S[key], resMC[key]))


        print('')
        print('FLUXES [Wm^-2]  6S    MC')
        for key in allkeys:
            print('{}  {:.3f}   {:.3f}'.format(key, ETOA_6S*res6S[key],
                                               ETOA_MC*resMC[key]))
        print('-----------------------------------------------------------------------')

        RES.iloc[count, RES.columns.get_loc('frac_abs_dir')] = resMC['frac_abs_dir']
        RES.iloc[count, RES.columns.get_loc('frac_abs_dif')] = resMC['frac_abs_dif']
        RES.iloc[count, RES.columns.get_loc('frac_abs_cou')] = resMC['frac_abs_cou']
        RES.iloc[count, RES.columns.get_loc('frac_abs_atm')] = resMC['frac_abs_atm']
        RES.iloc[count, RES.columns.get_loc('frac_toa_upw')] = resMC['frac_toa_upw']
        RES.iloc[count, RES.columns.get_loc('frac_abs_srf')] = resMC['frac_abs_srf']
        RES.iloc[count, RES.columns.get_loc('FTOAdown')] = ETOA_MC

        RES.iloc[ count + 1, RES.columns.get_loc('frac_abs_dir')] = res6S['frac_abs_dir']
        RES.iloc[ count + 1, RES.columns.get_loc('frac_abs_dif')] = res6S['frac_abs_dif']
        RES.iloc[ count + 1, RES.columns.get_loc('frac_abs_cou')] = res6S['frac_abs_cou']
        RES.iloc[ count + 1, RES.columns.get_loc('frac_abs_atm')] = res6S['frac_abs_atm']
        RES.iloc[ count + 1, RES.columns.get_loc('frac_toa_upw')] = res6S['frac_toa_upw']
        RES.iloc[ count + 1, RES.columns.get_loc('frac_abs_srf')] = res6S['frac_abs_srf']
        RES.iloc[ count + 1, RES.columns.get_loc('FTOAdown')]= ETOA_6S

        RES.iloc[count, RES.columns.get_loc('model')] = 'MC'
        RES.iloc[count + 1, RES.columns.get_loc('model')] = '6S'
        RES.iloc[count, RES.columns.get_loc('cosz')] = mycosz
        RES.iloc[count + 1, RES.columns.get_loc('cosz')] = mycosz
        RES.iloc[count, RES.columns.get_loc('freq')] = myfreq
        RES.iloc[count + 1, RES.columns.get_loc('freq')] = myfreq
        RES.iloc[count, RES.columns.get_loc('lambda')] =  mylambda
        RES.iloc[count + 1, RES.columns.get_loc('lambda')] =  mylambda
        count +=2

        # RES['frac_abs_dir'].iloc[count] = resMC['frac_abs_dir']
        # RES['frac_abs_dif'].iloc[count] = resMC['frac_abs_dif']
        # RES['frac_abs_cou'].iloc[count] = resMC['frac_abs_cou']
        # RES['frac_abs_atm'].iloc[count] = resMC['frac_abs_atm']
        # RES['frac_toa_upw'].iloc[count] = resMC['frac_toa_upw']
        # RES['frac_abs_srf'].iloc[count] = resMC['frac_abs_srf']
        # RES['FTOAdown'].iloc[count] = ETOA_MC
        # RES['frac_abs_cou'].iloc[count] = resMC['frac_abs_cou']

        # RES['frac_abs_dir'].iloc[count + 1] = res6S['frac_abs_dir']
        # RES['frac_abs_dif'].iloc[count + 1] = res6S['frac_abs_dif']
        # RES['frac_abs_cou'].iloc[count + 1] = res6S['frac_abs_cou']
        # RES['frac_abs_atm'].iloc[count + 1] = res6S['frac_abs_atm']
        # RES['frac_toa_upw'].iloc[count + 1] = res6S['frac_toa_upw']
        # RES['frac_abs_srf'].iloc[count + 1] = res6S['frac_abs_srf']
        # RES['FTOAdown'].iloc[count +1] = ETOA_6S

        # RES['model'].iloc[count] = 'MC'
        # RES['model'].iloc[count + 1] = '6S'
        # RES['cosz'].iloc[count] = mycosz
        # RES['cosz'].iloc[count + 1] = mycosz
        # RES['freq'].iloc[count] = myfreq
        # RES['freq'].iloc[count + 1] = myfreq
        # RES['lambda'].iloc[count] =  mylambda
        # RES['lambda'].iloc[count + 1] =  mylambda
        # count +=2

        # sys.stdout = orig_stdout
        # f.close()
        # print('surface impacts = ', eTOA/nphotons)

        # plt.figure()
        # # plt.plot(dfatm['wc'], dfatm['zz'][1:], 'o')
        # plt.plot(dfatm['extb'], dfatm['zz'][1:], 'o')
        # plt.show()
        #
        # dfatm2 = dfatm.copy()
        # dfatm2['zz'] = dfatm2['zz'][1:]
        # dfatm3 = pd.DataFrame(dfatm2)

        # plt.figure()
        # plt.plot(dfatm['extb'], dfatm['zz'][1:], 'o')
        # plt.plot(dfatm['absb'], dfatm['zz'][1:], 'o')
        # plt.plot(dfatm['scab'], dfatm['zz'][1:], 'o')
        # plt.show()

# R6S = RES[RES['model']=='6S'].copy()
# RMC = RES[RES['model']=='MC'].copy()

R6S = RES[(RES['model']=='6S') & (RES['cosz']==cosz2plot)].copy()
RMC = RES[(RES['model']=='MC') & (RES['cosz']==cosz2plot)].copy()


fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (15, 20))
msize = 20

axes[0, 0].set_title('Total direct flux absorbed surf.')
axes[0, 0].plot(R6S['freq'], R6S['frac_abs_dir']*R6S['FTOAdown'], 'o', markersize = msize, label='6S')
axes[0, 0].plot(R6S['freq'], RMC['frac_abs_dir']*RMC['FTOAdown'], '*', markersize = msize, label='RMC')
axes[0, 0].set_xticks(RMC['freq'])
axes[0, 0].set_xticklabels(RMC['freq'], rotation = 70)
axes[0, 0].set_xlabel(r'$\lambda$ $[\mathrm{\mu m}]$')
axes[0, 0].set_ylabel(r'Flux [W $\mathrm{m}^{-2} \mathrm{\mu m}^{-1}$]')
axes[0, 0].xaxis.set_label_coords(-0.10, -0.05)
axes[0, 0].legend()

axes[1, 0].set_title('Total diffuse flux absorbed surf.')
axes[1, 0].plot(R6S['freq'], R6S['frac_abs_dif']*R6S['FTOAdown'], 'o', markersize = msize, label='6S')
axes[1, 0].plot(R6S['freq'], RMC['frac_abs_dif']*RMC['FTOAdown'], '*', markersize = msize, label='RMC')
axes[1, 0].set_xticks(RMC['freq'])
axes[1, 0].set_xticklabels(RMC['lambda'], rotation = 70)
axes[1, 0].set_xlabel(r'$\lambda$ $[\mathrm{\mu m}]$')
axes[1, 0].set_ylabel(r'Flux [W $\mathrm{m}^{-2} [\mathrm{\mu m}^{-1}$]')
axes[1, 0].xaxis.set_label_coords(-0.10, -0.05)
axes[1, 0].legend()

axes[2, 0].set_title('Total coupled flux absorbed surf.')
axes[2, 0].plot(R6S['freq'], R6S['frac_abs_cou']*R6S['FTOAdown'], 'o', markersize = msize, label='6S')
axes[2, 0].plot(R6S['freq'], RMC['frac_abs_cou']*RMC['FTOAdown'], '*', markersize = msize, label='RMC')
axes[2, 0].set_xticks(RMC['freq'])
axes[2, 0].set_xticklabels(RMC['lambda'], rotation = 70)
axes[2, 0].set_xlabel(r'$\lambda$ $[\mathrm{\mu m}]$')
axes[2, 0].set_ylabel(r'Flux [W $\mathrm{m}^{-2} \mathrm{\mu m}^{-1}$]')
axes[2, 0].xaxis.set_label_coords(-0.10, -0.05)
axes[2, 0].legend()

axes[0, 1].set_title('Total flux up leaving TOA')
axes[0, 1].plot(R6S['freq'], R6S['frac_toa_upw']*R6S['FTOAdown'], 'o', markersize = msize, label='6S')
axes[0, 1].plot(R6S['freq'], RMC['frac_toa_upw']*RMC['FTOAdown'], '*', markersize = msize, label='RMC')
axes[0, 1].set_xticks(RMC['freq'])
axes[0, 1].set_xticklabels(RMC['lambda'], rotation = 70)
axes[0, 1].set_xlabel(r'$\lambda$ $[\mathrm{\mu m}]$')
axes[0, 1].set_ylabel(r'Flux [W $\mathrm{m}^{-2} \mathrm{\mu m}^{-1}$]')
axes[0, 1].xaxis.set_label_coords(-0.10, -0.05)
axes[0, 1].set_yscale('log')
axes[0, 1].legend()

axes[1, 1].set_title('Total flux absorbed by atmosph.')
axes[1, 1].plot(R6S['freq'], R6S['frac_abs_atm']*R6S['FTOAdown'], 'o', markersize = msize, label='6S')
axes[1, 1].plot(R6S['freq'], RMC['frac_abs_atm']*RMC['FTOAdown'], '*', markersize = msize, label='RMC')
axes[1, 1].set_xticks(RMC['freq'])
axes[1, 1].set_xticklabels(RMC['lambda'], rotation = 70)
axes[1, 1].set_xlabel(r'$\lambda$ $[\mathrm{\mu m}]$')
axes[1, 1].set_ylabel(r'Flux [W $\mathrm{m}^{-2} \mathrm{\mu m}^{-1}$]')
axes[1, 1].xaxis.set_label_coords(-0.10, -0.05)
# axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()

axes[2, 1].set_title('Total flux absorbed at surface')
axes[2, 1].plot(R6S['lambda'], R6S['FTOAdown'], '-*k', label='TOA Flux', markersize = msize)
axes[2, 1].plot(R6S['lambda'], R6S['frac_abs_srf']*R6S['FTOAdown'], 'o', markersize = msize, label='6S')
axes[2, 1].plot(R6S['lambda'], RMC['frac_abs_srf']*RMC['FTOAdown'], '*', markersize = msize, label='RMC')
xlabs = [0.35, 0.7, 1.0, 1.5, 2, 2.5,  3, 3.5]
axes[2, 1].set_xticks(xlabs)
axes[2, 1].set_xticklabels(xlabs, rotation = 90)
# axes[2, 1].set_xticks(RMC['lambda'])
# axes[2, 1].set_xticklabels(RMC['lambda'], rotation = 90)
axes[2, 1].set_xlabel(r'$\lambda$ $[\mathrm{\mu m}]$')
# axes[2, 1].set_xlabel(r'$\lambda [\mathrm{\mu m}]$')
axes[2, 1].set_ylabel(r'Flux [W $\mathrm{m}^{-2} \mathrm{\mu m}^{-1}$]')
axes[2, 1].xaxis.set_label_coords(-0.10, -0.05)
axes[2, 1].legend()
plt.tight_layout()
fig.savefig( os.path.join(outfigdir, 'valid_aero_{}_cosz_{}.png'.format(aerosol, cosz2plot)), dpi = 300)
plt.show()
