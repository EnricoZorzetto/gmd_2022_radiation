import os
import numpy as np
import pandas as pd
import  xarray as xr
import matplotlib.pyplot as plt
from geospatialtools import gdal_tools
import demfuncs as dem
import gridfuncs as gf
import matplotlib

from matplotlib import cm
from matplotlib.ticker import LinearLocator

########################## my parameters #################################
# domains = ['EastAlps, Nepal', 'Peru']
npvalues = [1, 2, 5, 10, 20, 100, 200]
# my_input_kind = 'k2n1pV'  # 2 hillslopes, 1 HB, variable # of tiles
my_input_kind = 'kVn1p5'  # 2 hillslopes, 1 HB, variable # of tiles
# inputkinds = ['kVn1p5']

myx= 1 # npvalue for the V variable
myk, myn, myp, varyingpar = gf.npk_from_string(
    my_input_kind=my_input_kind, myx=myx)
# myk = my_input_kind[1]
# myn = my_input_kind[3]
mydom = 'EastAlps'
# mydom = 'EastAlps'
# mydom = 'Peru'
# mycosz = 0.3
# myazi =    0.5*np.pi
myazi =    0.0*np.pi
myalbedo = 0.3

# directory to search for saved models (rfr/mlr)
# prefit_model_adir = '0.3'
# modeldir = os.path.join('/Users', 'ez6263', 'Documents',
#                         'dem_datasets', 'res_Peru_vs_EastAlps',
#                         'trained_models',
#                         'domain_EastAlps_buffer_0.1',
#                         'models_ave_6',
#                         )


prefit_model_aveblock = 1
prefit_avecorrection_aveblock = 110
prefit_model_buffer = 0.2
# prefit_model_buffer = 0.35
prefit_model_adir = 0.3
prefit_model_domain = 'Peru'
# modeldir = os.path.join('/Users', 'ez6263', 'Documents',
#                         'dem_datasets', 'res_Peru_vs_EastAlps',
#                         'trained_models',
#                         'domain_EastAlps_buffer_{}'.format(prefit_model_buffer),
#                         'models_ave_{}'.format(prefit_model_aveblock),
#                         )


SPECIFIC_PREDS = {
    'fdir': ['sian', 'svfn'],
    'fdif': ['sian', 'svfn', 'elen'],
    'frdir': ['sian','svfn', 'tcfn'],
    'frdif': ['svfn', 'tcfn'],
    'frdirn': ['sian','svfn', 'tcfn'],
    'frdifn': ['svfn', 'tcfn'],
    'fcoup': ['sian', 'svfn', 'tcfn'],
    'fcoupn': ['sian', 'svfn', 'tcfn']
}

parent_datadir = os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_data')
parent_outdir = os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_output')
modeldir = os.path.join( parent_outdir,
                         'res_Peru_vs_EastAlps',
                         # 'res_Peru_vs_EastAlps_buffer01',
                         'trained_models',
                         'domain_EastAlps_buffer_{}'.format(prefit_model_buffer),
                         'models_ave_{}'.format(prefit_model_aveblock),
                         )

modeldir_avecorrection = os.path.join( parent_outdir,
                         'res_Peru_vs_EastAlps',
                         # 'res_Peru_vs_EastAlps_buffer01',
                         'trained_models',
                         'domain_EastAlps_buffer_{}'.format(prefit_model_buffer),
                         'models_ave_{}'.format(prefit_avecorrection_aveblock),
                         )
########################################################################


#### LOAD RMC SIMUL
datadir_rmc = os.path.join('/Users', 'ez6263', "Documents", "gmd_2021", "gmd_2021_data",
                           "output_cluster_PP3D_EastAlps_merged123456")

FTOAnorm = 1
do_average = False
aveblock = 6
crop_buffer = 0.1

dem.matplotlib_update_settings()
# read tiling maps from GFDL_preprocessing codes
# datadir = os.path.join('/Users/ez6263/Documents/gmd_2021_grids_light')
datadir = os.path.join(parent_datadir, 'gmd_2021_grids_light')
# datadir = os.path.join(parent_datadir, 'gmd_2021_grids_light_NO_ELEN')
input_kinds = [f for f in os.listdir(datadir) if not f.startswith('.')]
ftilename = 'res_{}_k_{}_n_{}_p_{}'.format(mydom, myk, myn, myp)
myexpdir = os.path.join(datadir, my_input_kind, ftilename)
landdir = os.path.join(myexpdir, 'land', 'tile:1,is:1,js:1')
os.listdir(landdir)
dbfile = os.path.join(myexpdir, 'ptiles.{}.tile1.h5'.format(ftilename))
# gtd = gf.grid_tile_database(dbfile)
gtd = gf.grid_tile_database(dbfile, landdir=landdir, tiles2map=True)  # include high res maps




# import h5py
# dbs = h5py.File(dbfile, 'r')
# dbs['grid_data'].keys()
# dbs['grid_data']['tile:1,is:1,js:1']['lake'].keys()
# dbs['grid_data']['tile:1,is:1,js:1']['soil'].keys()
# dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_radstelev'][:]
# dbs['grid_data']['tile:1,is:1,js:1']['metadata'].keys()
# dbs['grid_data']['tile:1,is:1,js:1']['metadata']['latitude'][:]
# dbs['grid_data']['tile:1,is:1,js:1']['metadata']['wind'][:]
# elev1 = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_elevation'][:]
# frac1 = dbs['grid_data']['tile:1,is:1,js:1']['soil']['frac'][:]
# svf1 = dbs['grid_data']['tile:1,is:1,js:1']['soil']['tile_hlsp_svf'][:]
#
# np.sum(frac1*elev1)/np.sum(frac1)


COSZ = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
# COSZ = [0.1, 0.25]
# COSZ = [0.1, 0.25, 0.4, 0.55]
# COSZ = [0.1]
# COSZ = [0.1, 0.3, 0.5, 0.7, 0.85]
# COSZ = [0.1, 0.25, 0.4, 0.55]
ncosz = np.size(COSZ)
npoints = 6
nfluxes = 4
RES = np.zeros((ncosz, nfluxes, npoints))
RES_RMC = np.zeros((ncosz, nfluxes, 5))

# model_type = 'LEE'
model_type = 'MLR'
# model_type = 'NLR'
# model_type = 'RFR'
#

TOTFLUXES = np.zeros((ncosz, nfluxes))

for ic in range(ncosz):
    mycosz = COSZ[ic]



    # print('WARNING: reading simul for fixed cosz')
    # mycosz0 = 0.1
    res3d_ip = dem.load_3d_fluxes(FTOAnorm=FTOAnorm,
                                  cosz=mycosz,
                                  phi=myazi,
                                  adir=myalbedo,
                                  do_average=do_average,
                                  aveblock=aveblock,
                                  buffer=crop_buffer,
                                  datadir=datadir_rmc)

    respp_ip = dem.load_pp_fluxes(cosz=mycosz,
                                  adir=myalbedo,
                                  FTOAnorm=FTOAnorm,
                                  do_average=False,
                                  aveblock=1,
                                  buffer=crop_buffer,
                                  datadir=datadir_rmc)

    # respp_ip['elev']

    # rest_ip = dem.load_terrain_vars(cosz=mycosz, phi=myazi,
    #                                 buffer=crop_buffer,
    #                                 do_average=do_average,
    #                                 aveblock=aveblock,
    #                                 datadir=datadir_rmc)


    rest_ip = dem.load_static_terrain_vars(
                                    buffer=crop_buffer,
                                    do_average=do_average,
                                    aveblock=aveblock,
                                    datadir=datadir_rmc)
    rest_ip_dynamic = dem.comp_solar_incidence_field(rest_ip, cosz=mycosz, phi=myazi,
                        buffer=crop_buffer, do_average=do_average, aveblock=aveblock)
    rest_ip.update(rest_ip_dynamic)

    # np.mean(rest_ip['Z'])
    # plt.figure()
    # plt.imshow(rest_ip['Z'])
    # plt.show()


    fFMdir_field = (res3d_ip['FMdir'] - respp_ip['Fdir_pp']) / respp_ip['Fdir_pp']
    fFMdif_field = (res3d_ip['FMdif'] - respp_ip['Fdif_pp']) / respp_ip['Fdif_pp']
    fFMrdir_field = (res3d_ip['FMrdir']) / respp_ip['Fdir_pp']
    fFMrdif_field = (res3d_ip['FMrdif']) / respp_ip['Fdif_pp']
    fFMcoup_field = (res3d_ip['FMcoup'] - respp_ip['Fcoup_pp']) / respp_ip['Fcoup_pp']

    TOTFLUXES[ic, 0] = np.mean(res3d_ip['FMdir'])
    TOTFLUXES[ic, 1] = np.mean(res3d_ip['FMdif'])
    TOTFLUXES[ic, 2] = np.mean(res3d_ip['FMrdif'])
    TOTFLUXES[ic, 3] = np.mean(res3d_ip['FMrdir'])

    rmc_pred_aveflux = gf.prediction(svf=rest_ip['SVFnorm'], tcf=rest_ip['TCFnorm'], type=model_type, elen=rest_ip['elen'],
                              sian=rest_ip['SIAnorm'], cosz=mycosz, albedo=myalbedo,
                              modeldir = modeldir,
                                     specific_predictors=SPECIFIC_PREDS,
                                     prefit_models_adir = prefit_model_adir)


    rmc_pred_aveterrain = gf.prediction(svf=np.mean(rest_ip['SVFnorm']), tcf = np.mean(rest_ip['TCFnorm']), type=model_type,
                              sian=np.mean(rest_ip['SIAnorm']), cosz=mycosz, albedo=myalbedo, elen = np.mean(rest_ip['elen']),
                                modeldir = modeldir,
                                        specific_predictors=SPECIFIC_PREDS,
                                        prefit_models_adir = prefit_model_adir)


    flat_pred = gf.prediction(svf=1.0, tcf = 0.0, type=model_type,
                              sian=1.0, cosz=mycosz, albedo=myalbedo, elen = 0.0,
                              modeldir = modeldir,
                              specific_predictors=SPECIFIC_PREDS,
                              prefit_models_adir = prefit_model_adir)
    flat_pred_sian = gf.prediction(svf=1.0, tcf = 0.0, type=model_type, ss=0.0, sc = 0.0,
                                        sian=None, cosz=mycosz, azi = myazi, albedo=myalbedo, elen = 0.0,
                                        modeldir = modeldir,
                                   specific_predictors=SPECIFIC_PREDS,
                                   prefit_models_adir = prefit_model_adir)


    rmc_pred_aveflux.fdir_ave_pred = np.mean(rmc_pred_aveflux.fdir)
    rmc_pred_aveflux.fdif_ave_pred = np.mean(rmc_pred_aveflux.fdif)
    rmc_pred_aveflux.frdir_ave_pred = np.mean(rmc_pred_aveflux.frdir)
    rmc_pred_aveflux.frdif_ave_pred = np.mean(rmc_pred_aveflux.frdif)

    RES_RMC[ic, 0, 0] = np.mean(fFMdir_field) # RMC SIMULATION
    RES_RMC[ic, 1, 0] = np.mean(fFMdif_field)
    RES_RMC[ic, 2, 0] = np.mean(fFMrdir_field)
    RES_RMC[ic, 3, 0] = np.mean(fFMrdif_field)
    RES_RMC[ic, 0, 1] = rmc_pred_aveflux.fdir_ave_pred # RMC DATA, PREDICT HR and then average fluxes
    RES_RMC[ic, 1, 1] = rmc_pred_aveflux.fdif_ave_pred
    RES_RMC[ic, 2, 1] = rmc_pred_aveflux.frdir_ave_pred
    RES_RMC[ic, 3, 1] = rmc_pred_aveflux.frdif_ave_pred
    RES_RMC[ic, 0, 2] = rmc_pred_aveterrain.fdir # RMC DATA, first average terrain and then predict fluxes
    RES_RMC[ic, 1, 2] = rmc_pred_aveterrain.fdif
    RES_RMC[ic, 2, 2] = rmc_pred_aveterrain.frdir
    RES_RMC[ic, 3, 2] = rmc_pred_aveterrain.frdif

    RES_RMC[ic, 0, 3] = flat_pred.fdir # RMC DATA, first average terrain and then predict fluxes
    RES_RMC[ic, 1, 3] = flat_pred.fdif
    RES_RMC[ic, 2, 3] = flat_pred.frdir
    RES_RMC[ic, 3, 3] = flat_pred.frdif

    RES_RMC[ic, 0, 4] = flat_pred_sian.fdir # RMC DATA, first average terrain and then predict fluxes
    RES_RMC[ic, 1, 4] = flat_pred_sian.fdif
    RES_RMC[ic, 2, 4] = flat_pred_sian.frdir
    RES_RMC[ic, 3, 4] = flat_pred_sian.frdif

    ###### (0) predict average ucla fluxes - grid average [SCALAR]
    ucla_terrain = gf.ucla_terrain(domain=mydom)
    AVE_ucla_pred = gf.prediction(svf=ucla_terrain.svf, tcf=ucla_terrain.tcf, ss=ucla_terrain.ss, sc=ucla_terrain.sc,
                                  elen = 0.0,
                                  # type=model_type,
                                  type='LEE',
                                  cosz=mycosz, azi=myazi, albedo=myalbedo,
                                    modeldir = modeldir,
                                  specific_predictors=SPECIFIC_PREDS,
                                  prefit_models_adir = prefit_model_adir)


    print('ucla::')
    print(AVE_ucla_pred.sian)
    print(AVE_ucla_pred.fdir)
    print(AVE_ucla_pred.cosz)
    #   CZ FL NP
    RES[ic, 0, 0] = AVE_ucla_pred.fdir
    RES[ic, 1, 0] = AVE_ucla_pred.fdif
    RES[ic, 2, 0] = AVE_ucla_pred.frdir
    RES[ic, 3, 0] = AVE_ucla_pred.frdif

    ###### predict tile-by-tile fluxes [1D ARRAY]
    # tiles_pred = gf.prediction(svf=gtd.svf, tcf=gtd.tcf, ss=gtd.ss, sc=gtd.sc,
    #                            type='lee', cosz=mycosz, azi=myazi, albedo=myalbedo)

    tiles_pred = gf.prediction(svf=gtd.svf, tcf=gtd.tcf, ss=gtd.ss, sc=gtd.sc, elen=gtd.elen,
                               type=model_type, cosz=mycosz, azi=myazi, albedo=myalbedo,
                                modeldir = modeldir,
                               specific_predictors=SPECIFIC_PREDS,
                               prefit_models_adir = prefit_model_adir)
    ###### AVERAGE the tile-by-tile fluxes AFTER PREDICTION [SCALAR]
    tiles_pred.AVEfdir = np.sum(tiles_pred.fdir * gtd.frac)
    tiles_pred.AVEfdif = np.sum(tiles_pred.fdif * gtd.frac)
    tiles_pred.AVEfrdir = np.sum(tiles_pred.frdir * gtd.frac)
    tiles_pred.AVEfrdif = np.sum(tiles_pred.frdif * gtd.frac)

    # print( np.min( tiles_pred.fdif))
    # print( np.min( tiles_pred.frdir))
    # print( np.min( tiles_pred.frdif))

    #   CZ FL NP
    RES[ic, 0, 1] = tiles_pred.AVEfdir
    RES[ic, 1, 1] = tiles_pred.AVEfdif
    RES[ic, 2, 1] = tiles_pred.AVEfrdir
    RES[ic, 3, 1] = tiles_pred.AVEfrdif
    ####### predict tile-averaged fluxes [SCALAR] (First average terrain vars over tiles, then predict fluxes)
    AVE_tile_pred = gf.prediction(svf=gtd.ave_svf, tcf=gtd.ave_tcf, ss=gtd.ave_ss, sc=gtd.ave_sc, elen=gtd.ave_elen,
                                  type=model_type, cosz=mycosz, azi=myazi, albedo=myalbedo,
                                  # modeldir = modeldir,
                                  modeldir = modeldir_avecorrection, # since we are using areal average terrain, use large scale model
                                  specific_predictors=SPECIFIC_PREDS,
                                  prefit_models_adir = prefit_model_adir)
    #   CZ FL NP
    RES[ic, 0, 2] = AVE_tile_pred.fdir
    RES[ic, 1, 2] = AVE_tile_pred.fdif
    RES[ic, 2, 2] = AVE_tile_pred.frdir
    RES[ic, 3, 2] = AVE_tile_pred.frdif

    ###### predict high res map fluxes [2D MAPS] - add option here to compress them to 1D arrays?
    hrmap_pred = gf.prediction(svf=gtd.svf_hr_map.data, tcf=gtd.tcf_hr_map.data, elen=gtd.elen_hr_map.data,
                               ss=gtd.ss_hr_map.data, sc=gtd.sc_hr_map.data,
                               type=model_type, cosz=mycosz, azi=myazi, albedo=myalbedo,
                                modeldir = modeldir,
                               specific_predictors=SPECIFIC_PREDS,
                               prefit_models_adir = prefit_model_adir)

    ###### now, after perdiction average fluxes
    hrmap_pred.AVEfdir = np.mean(hrmap_pred.fdir)
    hrmap_pred.AVEfdif = np.mean(hrmap_pred.fdif)
    hrmap_pred.AVEfrdir = np.mean(hrmap_pred.frdir)
    hrmap_pred.AVEfrdif = np.mean(hrmap_pred.frdif)

    #   CZ FL NP
    RES[ic, 0, 3] = hrmap_pred.AVEfdir
    RES[ic, 1, 3] = hrmap_pred.AVEfdif
    RES[ic, 2, 3] = hrmap_pred.AVEfrdir
    RES[ic, 3, 3] = hrmap_pred.AVEfrdif
    ####### predict terrain-averaged fluxes [SCALAR] (First average tiles, then predict fluxes)
    AVE_hr_pred = gf.prediction(svf=gtd.svf_hr_ave, tcf=gtd.tcf_hr_ave, ss=gtd.ss_hr_ave, sc=gtd.sc_hr_ave, elen=gtd.elen_hr_ave,
                                type=model_type, cosz=mycosz, azi=myazi, albedo=myalbedo,
                                # modeldir = modeldir,
                                modeldir=modeldir_avecorrection, # since we are using areal average terrain, use large scale model
                                specific_predictors=SPECIFIC_PREDS,
                                prefit_models_adir = prefit_model_adir)
    # ucla_mydf = {'sian':AVE_ucla_pred.sian, 'tcfn':AVE_ucla_pred.tcf, 'svfn':AVE_ucla_pred.svf}
    # ucla_mydf = {'sian':AVE_ucla_pred.sian, 'tcfn':0.0, 'svfn':1.0}
    # AA = dem.lee_model_predict(ucla_mydf, label = 'fdir', cosz=mycosz, albedo=myalbedo)
    # print('ucla: cosz = {}, fdir = {}'.format(mycosz, AA))

    #   CZ FL NP
    RES[ic, 0, 4] = AVE_hr_pred.fdir
    RES[ic, 1, 4] = AVE_hr_pred.fdif
    RES[ic, 2, 4] = AVE_hr_pred.frdir
    RES[ic, 3, 4] = AVE_hr_pred.frdif



    ###### predict high res map fluxes [2D MAPS] - WITH CORRECTION BASED ON TERRAIN AVERAGE
    hrmap_pred_corr = gf.prediction(svf=gtd.svf_hr_map.data, tcf=gtd.tcf_hr_map.data, elen=gtd.elen_hr_map.data,
                               ss=gtd.ss_hr_map.data, sc=gtd.sc_hr_map.data,
                               type=model_type, cosz=mycosz, azi=myazi, albedo=myalbedo,
                               modeldir = modeldir, prefit_models_adir = prefit_model_adir,
                               normalize_by_grid_ave=True, normalize_fracs= None, # using 2d map not tiles here!
                                    specific_predictors=SPECIFIC_PREDS,
                                    normalize_grid_ave_modeldir=modeldir_avecorrection)

    ###### now, after perdiction average fluxes
    hrmap_pred_corr.AVEfdir = np.mean(hrmap_pred_corr.fdir)
    hrmap_pred_corr.AVEfdif = np.mean(hrmap_pred_corr.fdif)
    hrmap_pred_corr.AVEfrdir = np.mean(hrmap_pred_corr.frdir)
    hrmap_pred_corr.AVEfrdif = np.mean(hrmap_pred_corr.frdif)

    #   CZ FL NP
    RES[ic, 0, 5] = hrmap_pred_corr.AVEfdir
    RES[ic, 1, 5] = hrmap_pred_corr.AVEfdif
    RES[ic, 2, 5] = hrmap_pred_corr.AVEfrdir
    RES[ic, 3, 5] = hrmap_pred_corr.AVEfrdif

#
# import matplotlib
# matplotlib.use('MacOSX')
iff = 0  # fdir
plt.figure(figsize=(9, 9))
plt.plot(COSZ, RES[:, iff, 0], '--o', label="UCLA")
plt.plot(COSZ, RES[:, iff, 1], '-s', label="TILED - AVERAGE Fluxes" )
plt.plot(COSZ, RES[:, iff, 2], '-o', label="TILED - AVERAGE Terrain")
plt.plot(COSZ, RES[:, iff, 3], '-s', label="HR - AVERAGE Fluxes" )
# plt.plot(COSZ, RES[:, iff, 4], '--*', label="HR -- AVERAGE Terrain")
plt.plot(COSZ, RES[:, iff, 5], '-*', label="HR - AVERAGE Fluxes - with fdir ave correction" )
plt.plot(COSZ, np.zeros(ncosz), '-k')

plt.plot(COSZ, RES_RMC[:, iff, 0], '--^', label="RMC AVE SIMUL")
plt.plot(COSZ, RES_RMC[:, iff, 1], '-o', label="RMC PRED AVE FLUXES" )
plt.plot(COSZ, RES_RMC[:, iff, 2], '-*', label="RMC PRED AVE_TERRAIN" )
# plt.plot(COSZ, RES_RMC[:, iff, 3], '-^', label="CASE OF FLAT TERRAIN, SIAN GIVEN" )
# plt.plot(COSZ, RES_RMC[:, iff, 4], '--*', label="CASE OF FLAT TERRAIN, SS-SC GIVEN" )
plt.legend()
plt.show()

plt.figure()
plt.imshow(  hrmap_pred_corr.fdir, vmin=-0.5, vmax=0.5)
# plt.imshow(  hrmap_pred_corr.fdir)
plt.colorbar()
plt.show()

# # np.mean( hrmap_pred.fdir )
# np.mean(hrmap_pred.sian)
#
# plt.figure()
# plt.hist(  np.ravel(hrmap_pred.fdir), bins=40)
# plt.hist(  np.ravel(hrmap_pred_corr.fdir), bins=40, alpha=0.6)
# plt.xlim(-1.1,8)
# plt.show()

np.min(hrmap_pred_corr.fdir)
np.min(hrmap_pred.fdir)

# # hrmap_pred.fdir.shape
# NN = 100
# svfn = np.ones(NN)*0.9995
# tcfn = np.ones(NN)*0.0
# sian = np.linspace(0.9,1.1,NN)
# AA = {'svfn':svfn, 'sian':sian, 'tcfn':tcfn}
# res = dem.lee_model_predict(AA, label='fdir', cosz=0.1, albedo=0.3)
# print(res)
#
# plt.figure()
# plt.plot(sian, res)
# plt.show()
