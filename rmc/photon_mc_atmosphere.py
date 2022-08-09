import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import types
from numba.typed import Dict

from netCDF4 import Dataset
import Py6S
# from numba import jit

################################################################################

# Read a dictionary with optical properties of the atmosphere

# options: FuLiou, McClatchey, GFDL, Py6S

# variable needed for a N-layers atmosphere model:

# -> zz -> atmospheric levels (N+1)
# -> dz -> atmospheric layer thickness (N)
# -> tt -> atm optical depth of each single layer (N)
# -> extb -> atm total extinction coeff of each singler layer = tt/dz [m^-1]
# -> wc -> single scattering albedo for each layer (N)


################################################################################



def init_atmosphere_McClatchey():

    # Read McClatchey data for tropical atmosphere (TR)
    #                           clear aerosol distribution (CLEAR)
    #                           and wavelength 0.6934 microm
    # Note: coefficients are given in [km^-1]
    #       here must be converted to [m^-1]

    atm_data_folder = os.path.join('//', 'home', 'enrico', 'Documents',
                                   'dem_datasets')

    sfmc0 = pd.read_csv(os.path.join(atm_data_folder, 'McClatchey1971_TR_'
                                                     'CLEAR_06943.csv'))

    # sfmc0 = pd.read_csv(os.path.join(atm_data_folder, 'McClatchey1971_TR'
    #                                                   '_1536.csv'))
    sfmc = sfmc0.sort_values(by=['zz'], ascending=False)
    # ZZ -> lower bound of each layer (+ top of atm added here)
    # DZ -> thickness of each layer
    # ZZ = sfmc['zz']*1000.0 # convert to [m]
    nlayers = np.shape(sfmc)[0]
    nlevels = nlayers + 1
    dz     = sfmc['dz'].values*1000.0 # convert to [m]
    zz     = np.zeros(nlevels)
    zz[1:] = sfmc['zz'].values*1000.0
    zz[0] = zz[1] + dz[0]

    # EXTB = (sfmc['km'].values + sfmc['sigmam'].values)/1000.0 # convert to [m^-1]
    # SSA = sfmc['sigmam'].values/(sfmc['km'].values + sfmc['sigmam'].values + 1E-11)
    # TAU = EXTB*DZ
    #
    # ABSB = sfmc['km'].values/1000.0
    # SCAB = sfmc['sigmam'].values/1000.0

    k_abs_gas = sfmc['km'].values/1000.0
    k_abs_aer = sfmc['ka'].values/1000.0
    k_abs_tot = k_abs_gas + k_abs_aer
    k_sca_gas = sfmc['sigmam'].values/1000.0
    k_sca_aer = sfmc['sigmaa'].values/1000.0
    k_sca_tot = k_sca_gas + k_sca_aer

    k_ext_gas = k_abs_gas + k_sca_gas
    k_ext_aer = k_abs_aer + k_sca_aer
    k_ext_tot = k_ext_gas + k_ext_aer

    # single scattering albedos for gas and aerosol
    ssa_gas = k_sca_gas / (k_ext_gas + 1E-9)
    ssa_aer = k_sca_aer / (k_ext_aer + 1E-9)

    # asymmetry coefficient for aerosol scattering
    g_aer = np.ones(nlayers)*0.7


    dfatm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    dfatm["zz"] = np.asarray(zz, dtype=np.float64)
    dfatm["dz"] = np.asarray(dz, dtype=np.float64)
    dfatm["k_ext_tot"] = np.asarray(k_ext_tot, dtype=np.float64)
    dfatm["k_sca_tot"] = np.asarray(k_sca_tot, dtype=np.float64)
    dfatm["k_abs_tot"] = np.asarray(k_abs_tot, dtype=np.float64)
    dfatm["ssa_gas"] = np.asarray(ssa_gas, dtype=np.float64)
    dfatm["ssa_aer"] = np.asarray(ssa_aer, dtype=np.float64)
    dfatm["g_aer"] = np.asarray(g_aer, dtype=np.float64)
    dfatm["k_ext_gas"] = np.asarray(k_ext_gas, dtype=np.float64)
    dfatm["k_abs_gas"] = np.asarray(k_abs_gas, dtype=np.float64)
    dfatm["k_sca_gas"] = np.asarray(k_sca_gas, dtype=np.float64)
    dfatm["k_ext_aer"] = np.asarray(k_ext_aer, dtype=np.float64)
    dfatm["k_abs_aer"] = np.asarray(k_abs_aer, dtype=np.float64)
    dfatm["k_sca_aer"] = np.asarray(k_sca_aer, dtype=np.float64)

    return dfatm

# dfatm = init_atmosphere_McClatchey()

# dfatm['wc']

# dfdf = pd.DataFrame(dfatm)

def init_atmosphere_py6S(mylambda = None, cosz = None, aerosol = False):

    # cosz = 0.5
    # mylambda = 0.400
    # ALTITUDES = np.flipud(np.array([0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25,
    #                                 30, 35, 40, 50, 60, 100]))
    ALTITUDES = np.array([100, 70, 55, 50, 45, 40, 35, 30, 25, 23, 21, 20, 19,
            18 ,17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    # ALTITUDES = np.array([100, 70, 50, 30, 15, 0])
    # ALTITUDES = np.array([100, 20, 10, 7, 5, 3, 1.5, 0])
    # ALTITUDES = np.linspace(100, 0, 40)

    zz = ALTITUDES * 1000 # atmospheric levels in [m]
    nlevels = len(ALTITUDES)
    nlayers = nlevels - 1
    taucum_sca_gas = np.zeros(nlayers)
    taucum_abs_gas = np.zeros(nlayers)
    taucum_sca_aer = np.zeros(nlayers)
    taucum_abs_aer = np.zeros(nlayers)

    toa_flux = np.zeros(nlayers) # these should all be equal changing the z-level

    for i in range(nlayers):
        si = Py6S.SixS()
        myalti = ALTITUDES[i + 1]
        # print(myalti)

        # si.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Urban)
        # si.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Maritime)
        # si.aero_profile = Py6S.AeroProfile.NoAerosols
        # si.atmos_profile = AtmosProfile.PredefinedType(
        #     AtmosProfile.MidlatitudeSummer)

        if aerosol:
            si.aero_profile = Py6S.AeroProfile.PredefinedType(
                Py6S.AeroProfile.Maritime)
            # si.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Urban)
        else:
            si.aero_profile = Py6S.AeroProfile.NoAerosols

        # si.atmos_profile = Py6S.AtmosProfile.PredefinedType(
        #     Py6S.AtmosProfile.Tropical)
        si.atmos_profile = Py6S.AtmosProfile.PredefinedType(
            Py6S.AtmosProfile.MidlatitudeSummer)

        si.altitudes.set_target_custom_altitude(myalti)
        # si.altitudes.set_sensor_custom_altitude(0.00001)
        si.ground_reflectance = Py6S.GroundReflectance.HomogeneousLambertian(0.3)

        # Wavelength dependent values must be input at the following wavelengths
        # (given in micrometers):
        # ALL_LAMBDAS = np.array([0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515,
        #                     0.550, 0.590, 0.633, 0.670, 0.694, 0.760, 0.860,
        #                     1.240, 1.536, 1.650, 1.950, 2.250, 3.750])
        # nall_lambdas = len(ALL_LAMBDAS)
        # INDECES = np.arange(nall_lambdas)
        # lambda_dict = {}



        si.wavelength = Py6S.Wavelength(mylambda)
        # to get gaseous transmittance set sun at zenith
        # si.geometry.solar_z = np.arccos(cosz) * 180.0 / np.pi
        si.geometry.solar_z = 0.0
        si.run()

        # (0.00004*1 + 0.02997*0.29795 )/(0.00004 +0.02997 )


        # resdict = si.outputs.values
        # for ii in resdict.keys():
        #     print(ii)
        # res = si.outputs.values['single_scattering_albedo']

        # get absorption coeff for a vertical path downward:
        gas_abs_transm = si.outputs.total_gaseous_transmittance
        # taucum_abs_gas[i] = -np.log(gas_abs_transm)*cosz # if sun not at zenith
        taucum_abs_gas[i] = -np.log(gas_abs_transm)
        taucum_sca_gas[i] = si.outputs.optical_depth_total.rayleigh

        ssa_aer = si.outputs.single_scattering_albedo.aerosol
        opt_aer = si.outputs.optical_depth_total.aerosol

        # import numpy as np
        # import matplotlib.pyplot as plt
        # ZZ = np.linspace(100000.0, 0.0, 100)
        # Hscale = 30000 # hyp a 8000 m folding distance
        # ZEROLEVEL = 10000 # OD only above this layer
        # ODA = np.minimum(0.45*np.exp(-(ZZ[1:] - ZEROLEVEL)/Hscale), 0.45)
        # # ODA = 0.45*np.exp(-ZZ[1:]/Hscale)
        # plt.figure()
        # plt.plot(ODA, ZZ[1:], 'o')
        # plt.show()
        # assume exponential profile for aerosol optical depth conrtibutions:
        Hscale = 30000 # hyp a 8000 m folding distance
        ZEROLEVEL = 20000 # OD only above this layer
        # zz_mtop = zz[1:] # without TOA level
        # opt_aer_cumdistr = np.zeros(nlayers)
        opt_aer_cumdistr = np.minimum(opt_aer*np.exp(-(zz[1:] - ZEROLEVEL)/Hscale), opt_aer)
        taucum_sca_aer[i] = opt_aer_cumdistr[i]*ssa_aer
        taucum_abs_aer[i] = opt_aer_cumdistr[i]*(1-ssa_aer)

        # taucum_tot_sca[i] = si.outputs.optical_depth_total.total

        # taucum_gas_abs[i] = taucum_gas_tot[i] - taucum_gas_sca[i]

        # SSA[i] = si.outputs.single_scattering_albedo.rayleigh
        # TAUCUM[i] = taucum_gas_sca[i] + taucum_gas_abs[i]
        # print(si.outputs.fulltext)
        toa_flux[i] = si.outputs.solar_spectrum*cosz
        del si

    # print('taucum sca aer')
    # print(taucum_sca_aer)
    # print('taucum abs aer')
    # print(taucum_abs_aer)



    # COMPUTE LAYER-BY-LAYER OPTICAL DEPTHS FOR ATMOSPHERIC CONSTITUENTS
    tau_sca_aer = np.zeros(nlayers)
    tau_sca_aer[0] =  taucum_sca_aer[0]
    tau_sca_aer[1:] = taucum_sca_aer[1:] - taucum_sca_aer[:-1]

    tau_abs_aer = np.zeros(nlayers)
    tau_abs_aer[0] =  taucum_abs_aer[0]
    tau_abs_aer[1:] = taucum_abs_aer[1:] - taucum_abs_aer[:-1]

    tau_sca_gas = np.zeros(nlayers)
    tau_sca_gas[0] =  taucum_sca_gas[0]
    tau_sca_gas[1:] = taucum_sca_gas[1:] - taucum_sca_gas[:-1]

    tau_abs_gas = np.zeros(nlayers)
    tau_abs_gas[0] =  taucum_abs_gas[0]
    tau_abs_gas[1:] = taucum_abs_gas[1:] - taucum_abs_gas[:-1]

    tau_ext_aer = tau_sca_aer + tau_abs_aer # total extinction coefficient
    tau_ext_gas = tau_sca_gas + tau_abs_gas # total extinction coefficient
    tau_ext_tot = tau_ext_gas + tau_ext_aer # total extinction coefficient

    tau_abs_tot = tau_abs_gas + tau_abs_aer # total absorption coeff
    tau_sca_tot = tau_sca_gas + tau_sca_aer # total scattering coeff

    # compute scat/absorption/extinction coefficients for each layer
    # and for each atmospheric constituent (gas, aerosol, cloud)
    dz = zz[:-1] - zz[1:]

    k_abs_gas = tau_abs_gas/dz
    k_abs_aer = tau_abs_aer/dz
    # k_abs_aer = np.zeros(nlevels)
    k_abs_tot = tau_abs_tot/dz

    k_sca_gas = tau_sca_gas/dz
    k_sca_aer = tau_sca_aer/dz
    # k_sca_aer = np.zeros(nlayers)
    k_sca_tot = tau_sca_tot/dz

    k_ext_gas = tau_ext_gas/dz
    k_ext_aer = tau_ext_aer/dz
    # k_ext_aer = np.zeros(nlayers)
    k_ext_tot = tau_ext_tot/dz

    # single scattering albedos for gas and aerosol
    ssa_gas = k_sca_gas / (k_ext_gas + 1E-9)
    ssa_aer = k_sca_aer / (k_ext_aer + 1E-9)

    # asymmetry coefficient for aerosol scattering
    g_aer = np.ones(nlayers)*0.75


    dfatm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    dfatm["zz"] = np.asarray(zz, dtype=np.float64)
    dfatm["dz"] = np.asarray(dz, dtype=np.float64)
    dfatm["k_ext_tot"] = np.asarray(k_ext_tot, dtype=np.float64)
    dfatm["k_sca_tot"] = np.asarray(k_sca_tot, dtype=np.float64)
    dfatm["k_abs_tot"] = np.asarray(k_abs_tot, dtype=np.float64)
    dfatm["ssa_gas"] = np.asarray(ssa_gas, dtype=np.float64)
    dfatm["ssa_aer"] = np.asarray(ssa_aer, dtype=np.float64)
    dfatm["g_aer"] = np.asarray(g_aer, dtype=np.float64)
    dfatm["k_ext_gas"] = np.asarray(k_ext_gas, dtype=np.float64)
    dfatm["k_abs_gas"] = np.asarray(k_abs_gas, dtype=np.float64)
    dfatm["k_sca_gas"] = np.asarray(k_sca_gas, dtype=np.float64)
    dfatm["k_ext_aer"] = np.asarray(k_ext_aer, dtype=np.float64)
    dfatm["k_abs_aer"] = np.asarray(k_abs_aer, dtype=np.float64)
    dfatm["k_sca_aer"] = np.asarray(k_sca_aer, dtype=np.float64)
    dfatm['rsdcsaf'] = np.asarray(toa_flux, dtype = np.float64)

    return dfatm
#


def init_atmosphere_gfdl(metadata, myband = '16000_22650', mygpoint = 0):


    # myband = '16000_22650'
    # mygpoint = 1
    datadir = metadata['datadir']
    # atm_data_folder = os.path.join('//', 'home', 'enrico', 'Documents',
    #                                'dem_datasets')
    atm_data_folder = datadir
    atm_data = os.path.join(atm_data_folder, 'atmos_single_day.nc')

    ds = Dataset(atm_data, 'r')
    # ds['time'][:]

    # for keyi in list(ds.variables):
    #     print('__{}__'.format(keyi))
        # print(ds[keyi])
    clat = (metadata['maxlat'] + metadata['minlat']) / 2.0
    clon = (metadata['maxlon'] + metadata['minlon']) / 2.0

    xt = np.mean(ds['grid_xt_bnds'][:], axis=1)  # grid cell lon centers 0-360
    yt = np.mean(ds['grid_yt_bnds'][:], axis=1)  # grid cell lat centers -90 - 90

    myindx_x = np.argmin(np.abs(xt - clon))
    myindx_y = np.argmin(np.abs(yt - clat))

    # prm0 = ds['pressm']
    pr0 = ds['pressm'][:][0, :, myindx_y, myindx_x]

    tb_0 = ds['gas_shortwave_optical_depth_{}'.format(myband)]
    wb_0 = ds['gas_shortwave_single_scatter_albedo_{}'.format(myband)]
    gb_0 = ds['gas_shortwave_asymmetry_factor_{}'.format(myband)]

    rsdcsaf = ds['rsdcsaf'][0,:,myindx_y, myindx_x, mygpoint]
    # rsucsaf = ds['rsucsaf'][0,:,myindx_y, myindx_x, mygpoint]

    # print('GFDL BOA flux down = ', rsdcsaf[-1])
    # print('GFDL TOA flux down = ', rsdcsaf[0])
    # print('GFDL BOA flux fraction down = ', rsdcsaf[-1]/rsdcsaf[0])

    tt = tb_0[:][0, :, myindx_y, myindx_x, mygpoint]  # onhe value for each layer for each g point
    ssa_gas = wb_0[:][0, :, myindx_y, myindx_x, mygpoint]  # onhe value for each layer for each g point
    gas_ssca = ssa_gas
    g_gas = gb_0[:][0, :, myindx_y, myindx_x, mygpoint]  # onhe value for each layer for each g point
    # tt = np.fromfile( os.path.join(atmdir, 'tt'), dtype=float, sep = '\t') # optical depths [nlayers]
    # wc = np.fromfile( os.path.join(atmdir, 'wc'), dtype=float, sep = '\t') # single scatt. albedos [nlayers]
    # atmdir = 'sample_atm_profiles'
    # zz0 = np.fromfile( os.path.join(atmdir, 'zz'), dtype=float, sep = '\t') # elevation [nlevels]
    # # pr = np.fromfile( os.path.join(atmdir, 'pr'), dtype=float, sep = '\t') # pressure [nlevels]
    # mask_zz = np.ones(np.size(zz0), dtype=bool)
    # # mask_zz[1] = 0
    # mask_zz[-2] = 0
    # zz = zz0[mask_zz]



    dphalf = ds['dphalf'][:][0, :, myindx_y, myindx_x] # pressure intervals
    # dpflux = ds['dpflux'][:][0, :, myindx_y, myindx_x] # same as dphalf
    # phalf = ds['phalf'][:] # coordinate LEVELS (34,)
    # pfull = ds['pfull'][:] # coordinate LAYERS (33,)

    prcum = np.zeros(34)
    prcum[1:] = np.cumsum(dphalf)
    prcum = prcum + pr0[0] # add smallest value TOA
    pr = prcum
    # deltap0 = pr[-1] - prcum[-1]
    #

    # Compute elevation zz of each level based on hydrostatic pr. distribution
    To = 288.16 # sea level standard temperature [K]
    Ro = 8.314 # universal gas constant [J / mol / K]
    g = 9.806 # gravitational acceleration [m s^-2]
    # p0 = 101325 # standard sea level atm pressure [Pa]
    p0 = pr[-1] # standard sea level atm pressure [Pa] - use the zero level
    M = 0.0289 # kg/mol molar mass of dry air
    zz = To*Ro/g/M*np.log(p0/pr) # pr in [Pa]
    # dz = zz[:-1]-zz[1:]
    # print(pr)
    # print(pr.shape)


    # dz = To*Ro/g/M*np.log(pr[1:]/pr[:-1])



    # fix the lower boundary
    # zz[0] = 67000.0
    zz[-1] = 0.0
    dz = zz[:-1]-zz[1:]
    # extb[-1] = extb[-2]

    # print('zz', 'dz')
    # print(zz)
    # print(dz)

    k_ext_gas = tt/dz # volumetric extinction coefficient [m^-1]
    k_sca_gas = k_ext_gas*ssa_gas
    k_abs_gas = k_ext_gas*(1-ssa_gas)

    # ADD AEROSOL LATER
    # k_ext_tot = k_ext_gas
    # k_sca_tot = k_sca_gas
    # k_abs_tot = k_abs_gas

    # for debugging purposes, test enhanced scattering i.e., reduced opt. depth
    # if 'force_enhanced_extinction' in metadata.keys():
    #     if metadata['force_enhanced_extinction']:
    #         print('force a modified opt. cross section.')
    #         enh_factor = 1.0
    #         k_ext_gas = enh_factor*k_ext_gas
    #         k_sca_gas = enh_factor*k_sca_gas
    #         k_abs_gas = enh_factor*k_abs_gas

    #### EZDEV: ADDING AEROSOLS
    nlayers = np.size(dz)
    nlevels = nlayers + 1
    optical_band=myband
    not_quite_zero = 1E-9 # do divide by zero \o.0/
    #--------------- read optical properties of aerosol species ----------------
    AERS = [ 'dust1', 'dust2', 'dust3', 'dust4', 'dust5',
             'seasalt1', 'seasalt2', 'seasalt3', 'seasalt4', 'seasalt5',
             'bcphobic', 'omphobic', 'sulfate', 'volcanic']
    nae = len(AERS)

    AER_OPTD = np.zeros((nlayers, nae))
    AER_SSCA = np.zeros((nlayers, nae))
    AER_ASYM = np.zeros((nlayers, nae))

    # read aerosol optical properties
    for iae in range(nae):
        AER_OPTD[:,iae] = ds['{}_shortwave_optical_depth_{}'.format(
                  AERS[iae], optical_band)][:][
                          0, :, myindx_y, myindx_x]  # tau_i
        AER_SSCA[:,iae] = ds['{}_shortwave_single_scatter_albedo_{}'.format(
                  AERS[iae], optical_band)][:][
                          0, :, myindx_y, myindx_x]  # omega_i
        AER_ASYM[:,iae] = ds['{}_shortwave_asymmetry_factor_{}'.format(
                  AERS[iae], optical_band)][:][
                          0, :, myindx_y, myindx_x]  # g_i

    # average optical properties across aerosols
    aer_optd = np.sum(AER_OPTD, axis = 1)
    aer_ssca = np.sum(AER_OPTD*AER_SSCA, axis = 1)/(aer_optd + not_quite_zero)
    aer_asym = np.sum(AER_OPTD*AER_SSCA*AER_ASYM, axis = 1)/(aer_ssca + not_quite_zero)

    k_ext_aer = aer_optd/dz # volumetric extinction coefficient [m^-1]
    k_sca_aer = k_ext_aer*aer_ssca
    k_abs_aer = k_ext_aer*(1-aer_ssca)
    # --------------------------------------------------------------------------


    #--------------- read optical properties of clouds -------------------------
    CLOUDS = ['ice_cloud', 'liquid_cloud']
    ncl = len(CLOUDS)

    CLOUD_OPTD = np.zeros((nlayers, ncl))
    CLOUD_SSCA = np.zeros((nlayers, ncl))
    CLOUD_ASYM = np.zeros((nlayers, ncl))

    # read aerosol optical properties
    for icl in range(ncl):
        CLOUD_OPTD[:, icl] = ds['{}_shortwave_optical_depth_{}'.format(
            CLOUDS[icl], optical_band)][:][
                0, :, myindx_y, myindx_x]  # tau_i
        CLOUD_SSCA[:, icl] = ds['{}_shortwave_single_scatter_albedo_{}'.format(
            CLOUDS[icl], optical_band)][:][
                0, :, myindx_y, myindx_x]  # omega_i
        CLOUD_ASYM[:, icl] = ds['{}_shortwave_asymmetry_factor_{}'.format(
            CLOUDS[icl], optical_band)][:][
                0, :, myindx_y, myindx_x]  # g_i

    # average optical properties across aerosols
    cloud_optd = np.sum(CLOUD_OPTD, axis=1)
    cloud_ssca = np.sum(CLOUD_OPTD * CLOUD_SSCA, axis=1) / \
                 ( cloud_optd + not_quite_zero)
    cloud_asym = np.sum(CLOUD_OPTD * CLOUD_SSCA * CLOUD_ASYM, axis=1)/ \
                 ( cloud_ssca + not_quite_zero)

    k_ext_cloud = cloud_optd / dz  # volumetric extinction coefficient [m^-1]
    k_sca_cloud = k_ext_cloud * cloud_ssca
    k_abs_cloud = k_ext_cloud * (1 - cloud_ssca)

    #---------------------------------------------------------------------------


    # ############## TO ADD AER #########
    k_ext_tot = k_ext_gas + k_ext_aer
    k_sca_tot = k_sca_gas + k_sca_aer
    k_abs_tot = k_abs_gas + k_abs_aer
    # ###################################
    ##
    # NOTE:
    # To add clouds I would need to update also the
    # scattering code in the model,
    # if just updated here it would use the asymmetry param of aerosols
    ############## TO ADD AER  +  CLOUDS #########
    # k_ext_tot = k_ext_gas + k_ext_aer + k_ext_cloud
    # k_sca_tot = k_sca_gas + k_sca_aer + k_sca_cloud
    # k_abs_tot = k_abs_gas + k_abs_aer + k_abs_cloud
    ###################################

    # levels = np.arange(np.size(zz), 0, -1)  # level numbering


    # plt.figure()
    # # plt.plot(dphalf, 'ob')
    # # plt.plot(dpflux, '.r')
    # # plt.plot(pr, zz, 'o')
    # plt.plot(extb, zz[1:], 'o')
    # plt.show()
    # extb = extb*10

    # atm_dict = Dict.empty(
    #     key_type=types.unicode_type,
    #     value_type=types.float64[:],
    # )
    # atm_dict["rsdcsaf"] = np.asarray(rsdcsaf, dtype=np.float64)
    # atm_dict["rsucsaf"] = np.asarray(rsucsaf, dtype=np.float64)
    # atm_dict["zz"] = np.asarray(zz, dtype=np.float64)
    # atm_dict["pr"] = np.asarray(pr, dtype=np.float64)
    # atm_dict["tt"] = np.asarray(tt, dtype=np.float64)
    # atm_dict["gg"] = np.asarray(gg, dtype=np.float64)
    # atm_dict["wc"] = np.asarray(wc, dtype=np.float64)
    # atm_dict["extb"] = np.asarray(extb, dtype=np.float64)
    # atm_dict["scab"] = np.asarray(scab, dtype=np.float64)
    # atm_dict["absb"] = np.asarray(absb, dtype=np.float64)
    # atm_dict["dz"] = np.asarray(dz, dtype=np.float64)
    # atm_dict["levels"] = np.asarray(levels, dtype=np.float64)


    dfatm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    dfatm["zz"] = np.asarray(zz, dtype=np.float64)
    dfatm["dz"] = np.asarray(dz, dtype=np.float64)
    dfatm["k_ext_tot"] = np.asarray(k_ext_tot, dtype=np.float64)
    dfatm["k_sca_tot"] = np.asarray(k_sca_tot, dtype=np.float64)
    dfatm["k_abs_tot"] = np.asarray(k_abs_tot, dtype=np.float64)
    dfatm["ssa_gas"] = np.asarray(ssa_gas, dtype=np.float64)
    # dfatm["ssa_aer"] = np.asarray(ssa_aer, dtype=np.float64)
    # dfatm["g_aer"] = np.asarray(g_aer, dtype=np.float64)
    dfatm["k_ext_gas"] = np.asarray(k_ext_gas, dtype=np.float64)
    dfatm["k_abs_gas"] = np.asarray(k_abs_gas, dtype=np.float64)
    dfatm["k_sca_gas"] = np.asarray(k_sca_gas, dtype=np.float64)

    dfatm["rsdcsaf"] = np.asarray(rsdcsaf, dtype=np.float64)
    # dfatm["k_ext_aer"] = np.asarray(k_ext_aer, dtype=np.float64)
    # dfatm["k_abs_aer"] = np.asarray(k_abs_aer, dtype=np.float64)
    # dfatm["k_sca_aer"] = np.asarray(k_sca_aer, dtype=np.float64)
    # aer extinction coeffs
    dfatm["k_ext_aer"] = np.asarray(k_ext_aer, dtype=np.float64)
    dfatm["k_abs_aer"] = np.asarray(k_abs_aer, dtype=np.float64)
    dfatm["k_sca_aer"] = np.asarray(k_sca_aer, dtype=np.float64)

    # clouds extinction coeffs
    dfatm["k_ext_cloud"] = np.asarray(k_ext_cloud, dtype=np.float64)
    dfatm["k_abs_cloud"] = np.asarray(k_abs_cloud, dtype=np.float64)
    dfatm["k_sca_cloud"] = np.asarray(k_sca_cloud, dtype=np.float64)


    # single scattering albedos
    dfatm["gas_ssca"] = np.asarray(gas_ssca, dtype=np.float64)
    dfatm["aer_ssca"] = np.asarray(aer_ssca, dtype=np.float64)
    dfatm["cloud_ssca"] = np.asarray(cloud_ssca, dtype=np.float64)

    # asymmetry coefficients (skip gas, identically zero for Rayleigh scatt!)
    # dfatm["g_gas"] = np.asarray(gas_asym, dtype=np.float64)
    dfatm["g_aer"] = np.asarray(aer_asym, dtype=np.float64)
    dfatm["g_cloud"] = np.asarray(cloud_asym, dtype=np.float64)

    return dfatm




def init_atmosphere_gfdl_singletimestep(
        metadata, myband = '16000_22650', mygpoint = 0, zbar = 0.0):

    """-------------------------------------------------------------------------
    Read a single timestep dataset extracted from GFDL AM4
    To be used for validation
    2 bands are available: '16000_22650' or '12850_16000'
    Extract cosz and ground albedo from same location for validation for
    validation purposes

    -------------------------------------------------------------------------"""
    assert myband in ['16000_22650', '12850_16000'], \
        "GFDL atmosphere: provide valid band value!"

    assert mygpoint in list(np.arange(32)), \
        "GFDL atmosphere: provide valid gpoint value!"

    # NOTE: this only works for the specific dataset, change as necessary ------
    if mygpoint > 15:
        optical_band = '16000_22650'
        # optical_gpoint = mygpoint - 16
    else:
        optical_band = '12850_16000'
        # optical_gpoint = mygpoint
        # ----------------------------------------------------------------------

    not_quite_zero = 1E-9 # do divide by zero \o.0/

    datadir = metadata['datadir']
    # atm_data_folder = os.path.join('//', 'home', 'enrico', 'Documents',
    #                                'dem_datasets')
    atm_data_folder = datadir
    # atm_data = os.path.join(atm_data_folder, 'atmos_single_day.nc')
    atm_data = os.path.join(atm_data_folder, 'single-time-step.nc')

    ds = Dataset(atm_data, 'r')
    # ds['time'][:]

    # for keyi in list(ds.variables):
    #     print('__{}__'.format(keyi))
    # print(ds[keyi])
    clat = (metadata['maxlat'] + metadata['minlat']) / 2.0
    clon = (metadata['maxlon'] + metadata['minlon']) / 2.0

    xt = np.mean(ds['grid_xt_bnds'][:], axis=1)  # grid cell lon centers 0-360
    yt = np.mean(ds['grid_yt_bnds'][:], axis=1)  # grid cell lat centers -90 - 90

    myindx_x = np.argmin(np.abs(xt - clon))
    myindx_y = np.argmin(np.abs(yt - clat))

    # prm0 = ds['pressm']
    pr0 = ds['pressm'][:][0, :, myindx_y, myindx_x]



    rsdcsaf = ds['rsdcsaf'][0, mygpoint, :,myindx_y, myindx_x]
    # rsucsaf = ds['rsucsaf'][0,:,myindx_y, myindx_x, mygpoint]

    # print('GFDL BOA flux down = ', rsdcsaf[-1])
    # print('GFDL TOA flux down = ', rsdcsaf[0])
    # print('GFDL BOA flux fraction down = ', rsdcsaf[-1]/rsdcsaf[0])


    # compute elevation level starting from ground level
    dz = np.ravel(ds['dz'][:][0, :, myindx_y, myindx_x]) # pressure intervals
    nlayers = np.size(dz)
    nlevels = nlayers + 1
    zz = np.ones(nlevels)*zbar # start with average surface elevation
    for i in range(2, nlevels + 1):
        # print(i)
        zz[nlevels - i] = zz[nlevels - i + 1] + dz[nlevels - i]

    dphalf = ds['dphalf'][:][0, :, myindx_y, myindx_x] # pressure intervals
    # dpflux = ds['dpflux'][:][0, :, myindx_y, myindx_x] # same as dphalf
    # phalf = ds['phalf'][:] # coordinate LEVELS (34,)
    # pfull = ds['pfull'][:] # coordinate LAYERS (33,)



    prcum = np.zeros(nlevels)
    prcum[1:] = np.cumsum(dphalf)
    prcum = prcum + pr0[0] # add smallest value TOA
    pr = prcum
    # deltap0 = pr[-1] - prcum[-1]
    #

    # Compute elevation zz of each level based on hydrostatic pr. distribution
    To = 288.16 # sea level standard temperature [K]
    Ro = 8.314 # universal gas constant [J / mol / K]
    g = 9.806 # gravitational acceleration [m s^-2]
    # p0 = 101325 # standard sea level atm pressure [Pa]
    p0 = pr[-1] # standard sea level atm pressure [Pa] - use the zero level
    M = 0.0289 # kg/mol molar mass of dry air
    zz_hydrostatic = To*Ro/g/M*np.log(p0/pr) # pr in [Pa]

    # correct surface value:
    zz = zz + zz_hydrostatic[-1] # add ground level elevation

    # dz = zz[:-1]-zz[1:]
    # print(pr)
    # print(pr.shape)

    # print(zz)
    # print(zz_hydrostatic)


    # dz = To*Ro/g/M*np.log(pr[1:]/pr[:-1])



    # fix the lower boundary now
    # zz[0] = 67000.0
    zz_hydrostatic[-1] = 0.0
    dz_hydrostatic = zz_hydrostatic[:-1]-zz_hydrostatic[1:]

    zz[-1] = 0.0
    dz[-1] = zz[-2] - zz[-1]

    # extb[-1] = extb[-2]

    # print('zz', 'dz')
    # print(zz)
    # print(dz)


    #--------------- read optical properties of gases --------------------------
    tb_0 = ds['gas_shortwave_optical_depth']
    wb_0 = ds['gas_shortwave_single_scatter_albedo']
    # gb_0 = ds['gas_shortwave_asymmetry_factor'] # identically zero

    tt = tb_0[:][0,mygpoint, :, myindx_y, myindx_x]
    gas_ssca = wb_0[:][0,mygpoint, :, myindx_y, myindx_x]
    # gas_asym = gb_0[:][0,mygpoint, :, myindx_y, myindx_x]  # identically zero

    k_ext_gas = tt/dz # volumetric extinction coefficient [m^-1]
    k_sca_gas = k_ext_gas*gas_ssca
    k_abs_gas = k_ext_gas*(1-gas_ssca)
    # --------------------------------------------------------------------------


    #--------------- read optical properties of aerosol species ----------------
    AERS = [ 'dust1', 'dust2', 'dust3', 'dust4', 'dust5',
             'seasalt1', 'seasalt2', 'seasalt3', 'seasalt4', 'seasalt5',
             'bcphobic', 'omphobic', 'sulfate', 'volcanic']
    nae = len(AERS)

    AER_OPTD = np.zeros((nlayers, nae))
    AER_SSCA = np.zeros((nlayers, nae))
    AER_ASYM = np.zeros((nlayers, nae))

    # read aerosol optical properties
    for iae in range(nae):
        AER_OPTD[:,iae] = ds['{}_shortwave_optical_depth_{}'.format(
                  AERS[iae], optical_band)][:][
                          0, :, myindx_y, myindx_x]  # tau_i
        AER_SSCA[:,iae] = ds['{}_shortwave_single_scatter_albedo_{}'.format(
                  AERS[iae], optical_band)][:][
                          0, :, myindx_y, myindx_x]  # omega_i
        AER_ASYM[:,iae] = ds['{}_shortwave_asymmetry_factor_{}'.format(
                  AERS[iae], optical_band)][:][
                          0, :, myindx_y, myindx_x]  # g_i

    # average optical properties across aerosols
    aer_optd = np.sum(AER_OPTD, axis = 1)
    aer_ssca = np.sum(AER_OPTD*AER_SSCA, axis = 1)/(aer_optd + not_quite_zero)
    aer_asym = np.sum(AER_OPTD*AER_SSCA*AER_ASYM, axis = 1)/(aer_ssca + not_quite_zero)

    k_ext_aer = aer_optd/dz # volumetric extinction coefficient [m^-1]
    k_sca_aer = k_ext_aer*aer_ssca
    k_abs_aer = k_ext_aer*(1-aer_ssca)
    # --------------------------------------------------------------------------


    #--------------- read optical properties of clouds -------------------------
    CLOUDS = ['ice_cloud', 'liquid_cloud']
    ncl = len(CLOUDS)

    CLOUD_OPTD = np.zeros((nlayers, ncl))
    CLOUD_SSCA = np.zeros((nlayers, ncl))
    CLOUD_ASYM = np.zeros((nlayers, ncl))

    # read aerosol optical properties
    for icl in range(ncl):
        CLOUD_OPTD[:, icl] = ds['{}_shortwave_optical_depth_{}'.format(
            CLOUDS[icl], optical_band)][:][
                0, :, myindx_y, myindx_x]  # tau_i
        CLOUD_SSCA[:, icl] = ds['{}_shortwave_single_scatter_albedo_{}'.format(
            CLOUDS[icl], optical_band)][:][
                0, :, myindx_y, myindx_x]  # omega_i
        CLOUD_ASYM[:, icl] = ds['{}_shortwave_asymmetry_factor_{}'.format(
            CLOUDS[icl], optical_band)][:][
                0, :, myindx_y, myindx_x]  # g_i

    # average optical properties across aerosols
    cloud_optd = np.sum(CLOUD_OPTD, axis=1)
    cloud_ssca = np.sum(CLOUD_OPTD * CLOUD_SSCA, axis=1) / \
                 ( cloud_optd + not_quite_zero)
    cloud_asym = np.sum(CLOUD_OPTD * CLOUD_SSCA * CLOUD_ASYM, axis=1)/ \
                 ( cloud_ssca + not_quite_zero)

    k_ext_cloud = cloud_optd / dz  # volumetric extinction coefficient [m^-1]
    k_sca_cloud = k_ext_cloud * cloud_ssca
    k_abs_cloud = k_ext_cloud * (1 - cloud_ssca)

    #---------------------------------------------------------------------------


    k_ext_tot = k_ext_gas + k_ext_aer
    k_sca_tot = k_sca_gas + k_sca_aer
    k_abs_tot = k_abs_gas + k_abs_aer

    # TODO: to include clouds, use the following instead:a ---------------------
    # k_ext_tot = k_ext_gas + k_ext_aer + k_ext_cloud
    # k_sca_tot = k_sca_gas + k_sca_aer + k_sca_cloud
    # k_abs_tot = k_abs_gas + k_abs_aer + k_abs_cloud
    # TODO ---------------------------------------------------------------------


    # plt.figure()
    # # plt.plot(dphalf, 'ob')
    # # plt.plot(dpflux, '.r')
    # # plt.plot(pr, zz, 'o')
    # plt.plot(extb, zz[1:], 'o')
    # plt.show()
    # extb = extb*10

    # atm_dict = Dict.empty(
    #     key_type=types.unicode_type,
    #     value_type=types.float64[:],
    # )
    # atm_dict["rsdcsaf"] = np.asarray(rsdcsaf, dtype=np.float64)
    # atm_dict["rsucsaf"] = np.asarray(rsucsaf, dtype=np.float64)
    # atm_dict["zz"] = np.asarray(zz, dtype=np.float64)
    # atm_dict["pr"] = np.asarray(pr, dtype=np.float64)
    # atm_dict["tt"] = np.asarray(tt, dtype=np.float64)
    # atm_dict["gg"] = np.asarray(gg, dtype=np.float64)
    # atm_dict["wc"] = np.asarray(wc, dtype=np.float64)
    # atm_dict["extb"] = np.asarray(extb, dtype=np.float64)
    # atm_dict["scab"] = np.asarray(scab, dtype=np.float64)
    # atm_dict["absb"] = np.asarray(absb, dtype=np.float64)
    # atm_dict["dz"] = np.asarray(dz, dtype=np.float64)
    # atm_dict["levels"] = np.asarray(levels, dtype=np.float64)


    dfatm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    dfatm["zz"] = np.asarray(zz, dtype=np.float64)
    # dfatm["zz_hydrostatic"] = np.asarray(zz_hydrostatic, dtype=np.float64)
    dfatm["dz"] = np.asarray(dz, dtype=np.float64)
    # dfatm["dz_hydrostatic"] = np.asarray(dz_hydrostatic, dtype=np.float64)

    # down shortwave radiative flux (for validation purposes only)
    dfatm["rsdcsaf"] = np.asarray(rsdcsaf, dtype=np.float64)

    # total (gas + aerosol) extinction coeffs
    dfatm["k_ext_tot"] = np.asarray(k_ext_tot, dtype=np.float64)
    dfatm["k_sca_tot"] = np.asarray(k_sca_tot, dtype=np.float64)
    dfatm["k_abs_tot"] = np.asarray(k_abs_tot, dtype=np.float64)

    # gas extinction coeffs
    dfatm["k_ext_gas"] = np.asarray(k_ext_gas, dtype=np.float64)
    dfatm["k_abs_gas"] = np.asarray(k_abs_gas, dtype=np.float64)
    dfatm["k_sca_gas"] = np.asarray(k_sca_gas, dtype=np.float64)

    # aer extinction coeffs
    dfatm["k_ext_aer"] = np.asarray(k_ext_aer, dtype=np.float64)
    dfatm["k_abs_aer"] = np.asarray(k_abs_aer, dtype=np.float64)
    dfatm["k_sca_aer"] = np.asarray(k_sca_aer, dtype=np.float64)

    # clouds extinction coeffs
    dfatm["k_ext_cloud"] = np.asarray(k_ext_cloud, dtype=np.float64)
    dfatm["k_abs_cloud"] = np.asarray(k_abs_cloud, dtype=np.float64)
    dfatm["k_sca_cloud"] = np.asarray(k_sca_cloud, dtype=np.float64)

    # single scattering albedos
    dfatm["gas_ssca"] = np.asarray(gas_ssca, dtype=np.float64)
    dfatm["aer_ssca"] = np.asarray(aer_ssca, dtype=np.float64)
    dfatm["cloud_ssca"] = np.asarray(cloud_ssca, dtype=np.float64)

    # asymmetry coefficients (skip gas, identically zero for Rayleigh scatt!)
    # dfatm["g_gas"] = np.asarray(gas_asym, dtype=np.float64)
    dfatm["g_aer"] = np.asarray(aer_asym, dtype=np.float64)
    dfatm["g_cloud"] = np.asarray(cloud_asym, dtype=np.float64)

    return dfatm


def init_atmosphere_SampleFuLiou(atmdir='sample_atm_profiles'):
    # out-of-date
    # read atm data for a single g-point of the Fu-Liou 1992 model
    # obtained from NASA - CERES
    tt = np.fromfile( os.path.join(atmdir, 'tt'), dtype=float, sep = '\t') # optical depths [nlayers]
    ssa_gas = np.fromfile( os.path.join(atmdir, 'wc'), dtype=float, sep = '\t') # single scatt. albedos [nlayers]
    zz = np.fromfile( os.path.join(atmdir, 'zz'), dtype=float, sep = '\t') # elevation [nlevels]
    pr = np.fromfile( os.path.join(atmdir, 'pr'), dtype=float, sep = '\t') # pressure [nlevels]
    # add: read data from csv instead
    # return dict with uniform values,
    # for levels-type variables, skip lower level value
    # levels = np.arange(np.size(tt), 0, -1)  # level numbering

    nlayers = np.size(tt)
    # nlevels = np.size(tt) + 1
    dz = np.zeros(nlayers)  # thickness of each layer
    k_ext_gas = np.zeros(nlayers)  # average extinction coeff contribution of each layer

    # extb[0] = tt[0]
    for i in range(1, nlayers):
        dz[i - 1] = zz[i - 1] - zz[i]
        k_ext_gas[i] = (tt[i] - tt[i - 1]) / dz[i - 1]
    # now fill in first and last value respectively
    k_ext_gas[0] = tt[0] / dz[0]
    dz[-1] = zz[-2] - zz[-1]
    # return zz, pr, tt, wc, extb, dz, levels

    k_sca_gas = ssa_gas * k_ext_gas
    k_abs_gas = (1 - ssa_gas) * k_ext_gas

    # no aerosols in this case:
    k_ext_tot = k_ext_gas
    k_abs_tot = k_abs_gas
    k_sca_tot = k_sca_gas



    dfatm = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    dfatm["zz"]     = np.asarray(zz       , dtype = np.float64)

    dfatm["ssa_gas"] = np.asarray(ssa_gas, dtype=np.float64)
    dfatm["k_ext_gas"] = np.asarray(k_ext_gas, dtype=np.float64)
    dfatm["k_abs_gas"] = np.asarray(k_abs_gas, dtype=np.float64)
    dfatm["k_sca_gas"] = np.asarray(k_sca_gas, dtype=np.float64)

    dfatm["k_ext_tot"] = np.asarray(k_ext_tot, dtype=np.float64)
    dfatm["k_abs_tot"] = np.asarray(k_abs_tot, dtype=np.float64)
    dfatm["k_sca_tot"] = np.asarray(k_sca_tot, dtype=np.float64)
    dfatm["dz"]     = np.asarray(dz       , dtype = np.float64)


    # dfatm["dz"]     = np.asarray(dz       , dtype = np.float64)

    return dfatm




def plot_atmospheric_profiles(dfatm):
    plt.figure()
    plt.plot(dfatm['k_ext_tot'], dfatm['zz'][1:], 'o')
    plt.plot(dfatm['k_ext_aer'], dfatm['zz'][1:], 'o')
    plt.plot(dfatm['k_ext_gas'], dfatm['zz'][1:], 'o')
    plt.plot(dfatm['k_ext_gas'] + dfatm['k_ext_aer'], dfatm['zz'][1:], '.')
    plt.show()

    plt.figure()
    plt.plot(dfatm['k_abs_tot'], dfatm['zz'][1:], 'o')
    plt.plot(dfatm['k_abs_aer'], dfatm['zz'][1:], '.')
    plt.plot(dfatm['k_abs_gas'], dfatm['zz'][1:], '.')
    plt.show()

    plt.figure()
    plt.plot(dfatm['k_sca_tot'], dfatm['zz'][1:], 'o')
    plt.plot(dfatm['k_sca_aer'], dfatm['zz'][1:], '.')
    plt.plot(dfatm['k_sca_gas'], dfatm['zz'][1:], '.')
    plt.plot(dfatm['k_sca_gas'] + dfatm['k_sca_aer'], dfatm['zz'][1:], '.')
    plt.show()
    return


# @jit(nopython=True)
def get_toa_up_irradiance_py6S(s, nazi = 16, nzen = 20):
    # provide a Py6S class in input
    phi_interval = 2*np.pi
    nphiintervals = nazi # 16
    dphi = phi_interval / nphiintervals
    PHIs = np.linspace(dphi/2, phi_interval - dphi/2, nphiintervals)


    theta_interval = np.pi/2
    nthetaintervals = nzen # 10
    dtheta = theta_interval / nthetaintervals
    # interval centers
    THETAs = np.linspace(dtheta/2, theta_interval - dtheta/2, nthetaintervals)
    # dtheta = THETAs[1]- THETAs[0]
    # nthetas = np.size(THETAs)
    ETOAup = 0
    for it in range(nthetaintervals):
        # print(it)
        mytheta_rad = THETAs[it]
        mytheta_deg = THETAs[it] * 180.0 / np.pi
        s.geometry.view_z = mytheta_deg
        for ip in range(nphiintervals):

            # myphi_rad = PHIs[ip]
            myphi_deg = PHIs[ip] * 180.0 / np.pi

            s.geometry.view_a = myphi_deg


            s.run()
            Lit = s.outputs.apparent_radiance
            # Lit = 1

            # radatm = s.outputs.atmospheric_intrinsic_radiance
            # radback = s.outputs.background_radiance
            # radapp = s.outputs.pixel_radiance
            # radapp = s.outputs.apparent_radiance
            # radtot = radatm + radback + radapp
            Eit = Lit*np.cos(mytheta_rad)*np.sin(mytheta_rad)*dtheta*dphi
            # print(it, Lit, Eit)
            ETOAup +=  Eit
    # ETOAup = ETOAup 2 * np.pi
    # print(ETOAup)
    return ETOAup

