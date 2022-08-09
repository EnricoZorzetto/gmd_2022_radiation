

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geospatialtools import gdal_tools
import demfuncs as dem
import gridfuncs as gf
import tilefuncs as til
import matplotlib
import xarray as xr
import string
import gdal_write_raster as rster

########################## my parameters #################################
# domains = ["EastAlps", "Peru"]
# domains = ["Peru"]
# domains = ["Nepal"]
# gridlist = {

domains = ["EastAlps", "Nepal", "Peru"]
# domains = ["EastAlps"]
# domains = ["EastAlps", "Nepal", "Peru"]

# domains = ["Nepal", "Peru"]
npvalues = [1, 2, 5, 10, 20, 50, 100, 200]
# npvalues = [1, 5, 50]
# npvalues = [5, 20, 100]

 # plot 3 of them - must be in npvalues - first value must be the 1st value in npvalues
# NPV_2_PLOT = [1, 2,5]
NPV_2_PLOT = [1, 5, 50]
# npvalues = [10, 200]

# domains = ["EastAlps"]
# npvalues = [ 2]
# K must be at least 2!
inputkinds = ['k2n1pV', 'k5n1pV', 'k2nVp2', 'kVn1p2', 'kVn1p5']
# inputkinds = ['k5n1pV','kVn1p5']
# inputkinds = ['k2n1pV','kVn1p2']
# inputkinds = ['k5n1pV']
# inputkinds = ['kVn1p2']
# my_input_kind = 'k2n1pV'; # 2 hillslopes, 1 HB, variable # of tiles
# myk=my_input_kind[1]; myn=my_input_kind[3]
# mydom = 'EastAlps'
# mydom = 'EastAlps'
# myp = 1
mycosz = 0.4
# myazi = -0.0*np.pi
myazi_index = 0.0
myazi = myazi_index*np.pi
myalbedo = 0.3

# by default, compare tiled statistics to those of the original SRTM field (90m)
# If instead we want to compare it to stats at a different aggregation scale, turn this True
# and select the size of the averaging block (e.g., hrmap_aveblock = 6 -> size = 90m * 6
do_average_hr_fields = False
hrmap_aveblock = 32

# prefit_model_aveblock = 1
prefit_model_aveblock = 6
prefit_avecorrection_aveblock = 55
# prefit_model_buffer = 0.35
prefit_model_buffer = 0.2 # turn to 0.1
prefit_model_adir = 0.3
# prefit_model_domain = 'EastAlps'
prefit_model_domain = 'Peru'


parent_datadir = os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_data')
parent_outdir = os.path.join('/Users', 'ez6263', 'Documents', 'gmd_2021', 'gmd_2021_output')


modeldir = os.path.join( parent_outdir,
                        'res_Peru_vs_EastAlps',
                        'trained_models',
                        'domain_EastAlps_buffer_{}'.format(prefit_model_buffer),
                        'models_ave_{}'.format(prefit_model_aveblock),
                        )

modeldir_avecorrection = os.path.join( parent_outdir,
                                       # 'res_Peru_vs_EastAlps',
                                       'res_Peru_vs_EastAlps',
                                       'trained_models',
                                       'domain_EastAlps_buffer_{}'.format(prefit_model_buffer),
                                       'models_ave_{}'.format(prefit_avecorrection_aveblock),
                                       )
########################################################################


# stats = ['stdv', 'skew', 'kurt']
stats = ['mean', 'stdv', 'skew', 'kurt']

# models = ['LEE', 'MLR', 'RFR', 'NLR']
models = ['MLR']

# labels = ['fdir', 'fdif', 'frdirn', 'frdifn','fcoupn']
# labels = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']
labels = ['fdir', 'fdif', 'frdir', 'frdif']
terrains = ['sc', 'ss', 'svf', 'tcf', 'elen']


# PREDICTORS_TO_USE = ['elen', 'sian', 'tcf0', 'svf0']
# SPECIFIC_PREDS = {
#     'fdir': ['sian', 'svf0'],
#     'fdif': ['sian', 'svf0', 'elen'],
#     'frdir': ['sian','svf0', 'tcf0'],
#     'frdif': ['svf0', 'tcf0'],
#     'frdirn': ['sian','svf0', 'tcf0'],
#     'frdifn': ['svf0', 'tcf0'],
#     'fcoup': ['sian', 'svf0', 'tcf0'],
#     'fcoupn': ['sian', 'svf0', 'tcf0']
# }



PREDICTORS_TO_USE = ['elen', 'sian', 'tcfn', 'svfn']
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



dsres = gf.init_metric_tiles(domains, npvalues, inputkinds, models, labels, terrains, stats,
                      prefit_model_aveblock=prefit_model_aveblock, prefit_model_buffer=prefit_model_buffer,
                      prefit_model_adir=prefit_model_adir, prefit_model_domain=prefit_model_domain,
                             cosz=mycosz, azi=myazi, adir=myalbedo)

dem.matplotlib_update_settings()
# read tiling maps from GFDL_preprocessing codes
datadir = os.path.join(parent_datadir, 'gmd_2021_grids_light')
# datadir = os.path.join(parent_datadir, 'gmd_2021_grids_light')
# datadir_pck = os.path.join('/Users/ez6263/Documents/gmd_2021_grids_light_pck')
outdir = os.path.join(parent_outdir, 'gmd_2021_grids_output')
outfigdir = os.path.join(parent_outdir, 'gmd_2021_grids_output','outfigdir')

conv_outfigdir = os.path.join(outfigdir, 'conv_plots_cosz_{}_adir_{}'.format(mycosz, myazi))
os.system("mkdir -p {}".format(outdir))
os.system("mkdir -p {}".format(outfigdir))
os.system("mkdir -p {}".format(conv_outfigdir))




for id, mydom in enumerate(domains):
    # outfigdir_id = os.path.join(outfigdir, mydom)
    outfigdir_id = os.path.join(conv_outfigdir, mydom)
    os.system("mkdir -p {}".format(outfigdir_id))
    outfigdir_ihr = os.path.join(outfigdir_id, "hr_fields")
    os.system("mkdir -p {}".format(outfigdir_ihr))
    for ik, my_input_kind in enumerate(inputkinds):

        # if ik == 0 and id == 0: # save high res maps only in this case
        RES_ALL = {mynp:None for mynp in npvalues} # init dict collecting grid objects

        outfigdir_ik = os.path.join(outfigdir_id, "{}_single_maps".format(my_input_kind))
        os.system("mkdir -p {}".format(outfigdir_ik))
        outfigdir_ik_rasters = os.path.join(outfigdir_id, "{}_rasters".format(my_input_kind))
        os.system("mkdir -p {}".format(outfigdir_ik_rasters))

        for ip, myx in enumerate(npvalues):
            print('reading results for domain = {}, kind = {}, npvalues = {}'.format(mydom, my_input_kind, myx))

            myk, myn, myp, varyingpar = gf.npk_from_string(
                        my_input_kind=my_input_kind, myx=myx)

            input_kinds =[f for f in os.listdir(datadir) if not f.startswith('.')]
            ftilename = 'res_{}_k_{}_n_{}_p_{}'.format(mydom, myk, myn, myp)
            myexpdir = os.path.join(datadir, my_input_kind, ftilename)
            landdir = os.path.join(myexpdir, 'land', 'tile:1,is:1,js:1')
            dbfile = os.path.join(myexpdir, 'ptiles.{}.tile1.h5'.format(ftilename))

            # myexpdir_pck = os.path.join(datadir_pck, my_input_kind, ftilename)
            # pckdir = os.path.join(myexpdir_pck, 'land', 'tile:1,is:1,js:1')

            gtd = gf.grid_tile_database(dbfile, landdir=landdir, tiles2map=True) # include high res maps
            # gtd = gf.grid_tile_database(dbfile, landdir=landdir, tiles2map=True, pckdir=None) # include high res maps
            # print("id = {}, ik = {}, ip = {}".format(id, ik, ip))
            # print("k={}, n={}, p={}, ntiles = {}".format(myk, myn, myp, gtd.ntiles))
            # print(gtd.ntiles)
            gtd.map_tiles_to_map() # get spatial distribution of tiles and properties
            gtd.remap_tiles_to_equal_area(outdir=outfigdir_ik_rasters) # here we are overwriting at each ip, but it is ok

            # vars_to_plot = ['svf_tiles', 'tcf_tiles', 'ss_tiles', 'sc_tiles', 'tiles']
            vars_to_plot = ['svf_tiles', 'tcf_tiles', 'ss_tiles', 'sc_tiles', 'tiles', 'elen_tiles']
            for myvar_tiled in vars_to_plot:
                outputdir1 = os.path.join(outfigdir_ik,
                        'tilemap_{}_{}_{}.png'.format(myvar_tiled, varyingpar, myx))
                gtd.plot_map_field(var=myvar_tiled, outputdir=outputdir1)

            # for plotting purposes - save entire list of objects
            RES_ALL[myx] = gtd

            hr_vars_to_plot = ['svf', 'tcf', 'ss', 'sc', 'elen']
            for myvar_hr in hr_vars_to_plot:

                # SAVE TILED TERRAIN STATISTICS ON XARRAY DATASET
                # stats4_tiled_terr = til.comp_4_stats(gtd.tiled_terrain[myvar_hr], weights=gtd.frac)
                stats4_tiled_terr = gf.comp_4_stats(gtd.tiled_terrain[myvar_hr], weights=gtd.frac)
                for mystat_tiled_terr in stats4_tiled_terr.keys():
                    dsres['tiled_terrains'].loc[dict(
                        stats=mystat_tiled_terr,
                        inputkinds=my_input_kind,
                        npvalues=myx,
                        terrains=myvar_hr,
                        domains=mydom
                    )] = stats4_tiled_terr[mystat_tiled_terr]

                # if ik == 0 and ip == 0: # save stats for high res fields
                if ip == 0: # save stats for high res fields
                    outputdir2 = os.path.join(outfigdir_ihr, '{}.png'.format(myvar_hr))
                    gtd.plot_map_field(var=myvar_hr, outputdir=outputdir2)

                    # SAVE HIGH RES STATISTICS ON XARRAY DATASET
                    # stats4_hires_terr = til.comp_4_stats(gtd.hr_terrain[myvar_hr].data.flatten())
                    stats4_hires_terr = gf.comp_4_stats(gtd.hr_terrain[myvar_hr].data.flatten())
                    for mystat_hr in stats4_hires_terr.keys():
                        dsres['hires_terrains'].loc[dict(
                                               stats=mystat_hr,
                                               terrains=myvar_hr,
                                               domains=mydom
                                               )] = stats4_hires_terr[mystat_hr]


            # compute tiled fluxes
            # for now let's try to add them to the gtd object
            for mymod in models:

                # if ik == 0 and ip == 0:  # save stats for high res fields
                if ip == 0:  # save stats for high res fields

                    if do_average_hr_fields:
                        aveblocked_svf = dem.crop_and_average(gtd.svf_hr_map.data, buffer=0, average=True, aveblock=hrmap_aveblock)
                        aveblocked_tcf = dem.crop_and_average(gtd.tcf_hr_map.data, buffer=0, average=True, aveblock=hrmap_aveblock)
                        aveblocked_ss = dem.crop_and_average(gtd.ss_hr_map.data, buffer=0, average=True, aveblock=hrmap_aveblock)
                        aveblocked_sc = dem.crop_and_average(gtd.sc_hr_map.data, buffer=0, average=True, aveblock=hrmap_aveblock)
                        aveblocked_elen = dem.crop_and_average(gtd.elen_hr_map.data, buffer=0, average=True, aveblock=hrmap_aveblock)

                        gtd.hires_fluxes = gf.prediction(svf=aveblocked_svf,
                                                         tcf=aveblocked_tcf,
                                                         ss=aveblocked_ss,
                                                         sc=aveblocked_sc,
                                                         elen = aveblocked_elen,
                                                         type = mymod, cosz = mycosz, azi = myazi,
                                                         albedo = myalbedo, sian=None,
                                                         modeldir = modeldir, prefit_models_adir = prefit_model_adir,
                                                         normalize_by_grid_ave=True, normalize_fracs=None, # using 2d map not tiles here!
                                                         specific_predictors=SPECIFIC_PREDS,
                                                         normalize_grid_ave_modeldir=modeldir_avecorrection)


                        # gtd.hires_corrected = gf.prediction(svf=aveblocked_svf,
                        #                                  tcf=aveblocked_tcf,
                        #                                  ss=aveblocked_ss,
                        #                                  sc=aveblocked_sc,
                        #                                  elen = aveblocked_elen,
                        #                                  type = mymod, cosz = mycosz, azi = myazi,
                        #                                  albedo = myalbedo, sian=None,
                        #                                  modeldir = modeldir, prefit_models_adir = prefit_model_adir
                        #                                     )


                    else:
                        gtd.hires_fluxes = gf.prediction(svf=gtd.svf_hr_map.data,
                                                         tcf=gtd.tcf_hr_map.data,
                                                         ss=gtd.ss_hr_map.data,
                                                         sc=gtd.sc_hr_map.data,
                                                         elen = gtd.elen_hr_map.data, # TODO add elen
                                                         type = mymod, cosz = mycosz, azi = myazi,
                                                         albedo = myalbedo, sian=None,
                                                         modeldir = modeldir, prefit_models_adir = prefit_model_adir,
                                                         normalize_by_grid_ave=True, normalize_fracs=None, # using 2d map not tiles here!
                                                         specific_predictors=SPECIFIC_PREDS,
                                                         normalize_grid_ave_modeldir=modeldir_avecorrection)

                    # SAVE HIGH RES STATISTICS ON XARRAY DATASET
                    # fluxes_to_save = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
                    fluxes_to_save = labels
                    for myflux in fluxes_to_save:
                        stats4_hires_flux = gf.comp_4_stats(gtd.hires_fluxes.fluxes[myflux])
                        for mystat_hr in stats4_hires_flux.keys():
                            dsres['hires_fluxes'].loc[dict(
                                stats=mystat_hr,
                                labels=myflux,
                                domains=mydom,
                                models=mymod
                            )] = stats4_hires_flux[mystat_hr]

                gtd.tiled_fluxes = gf.prediction(svf=gtd.svf, tcf=gtd.tcf, elen = gtd.elen, # TODO add elen
                                             ss=gtd.ss, sc=gtd.sc,
                                             type=mymod, cosz=mycosz, azi=myazi,
                                             albedo=myalbedo, sian=None,
                                             modeldir = modeldir, prefit_models_adir = prefit_model_adir,
                                             normalize_by_grid_ave = True, normalize_fracs = gtd.frac,  # using 2d map not tiles here!
                                                 specific_predictors=SPECIFIC_PREDS,
                                                 normalize_grid_ave_modeldir = modeldir_avecorrection)


                # print( 'min fdif =', np.min(gtd.tiled_fluxes.fdif))
                # print( 'min frdif =', np.min(gtd.tiled_fluxes.frdif))
                # print( 'min frdir =', np.min(gtd.tiled_fluxes.frdir))
                assert np.min(gtd.tiled_fluxes.fdif > -1.1)

                # # SAVE TILE - STATISTICS ON XARRAY DATASET
                # fluxes_to_save = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
                fluxes_to_save = labels
                for myflux in fluxes_to_save:
                    stats4_tiled_flux = gf.comp_4_stats(gtd.tiled_fluxes.fluxes[myflux], weights=gtd.frac)
                    for mystat_tiled in stats4_tiled_flux.keys():
                        dsres['tiled_fluxes'].loc[dict(
                            stats=mystat_tiled,
                            labels=myflux,
                            domains=mydom,
                            inputkinds=my_input_kind,
                            npvalues=myx,
                            models=mymod
                        )] = stats4_tiled_flux[mystat_tiled]


                gtd.mappedt_fluxes = gf.prediction(svf=gtd.svf_mapped_ea.data, tcf=gtd.tcf_mapped_ea.data, elen = gtd.elen_mapped_ea.data, # TODO add elen
                                                   ss=gtd.ss_mapped_ea.data, sc=gtd.sc_mapped_ea.data,
                                                   type=mymod, cosz=mycosz, azi=myazi,
                                                   albedo=myalbedo, sian=None,
                                                   specific_predictors=SPECIFIC_PREDS,
                                                   modeldir = modeldir, prefit_models_adir = prefit_model_adir)


                gtd.mappedtll_fluxes = gf.prediction(svf=gtd.svf_mapped, tcf=gtd.tcf_mapped, elen=gtd.elen_mapped, # TODO add elen
                                                   ss=gtd.ss_mapped, sc=gtd.sc_mapped,
                                                   type=mymod, cosz=mycosz, azi=myazi,
                                                   albedo=myalbedo, sian=None,
                                                     specific_predictors=SPECIFIC_PREDS,
                                                     modeldir = modeldir, prefit_models_adir = prefit_model_adir)

                # overwrite mapping tiled fluxes over tiles::
                # gtd.map_fluxes_to_map()

                # def map_tiled_prop(ptid=None, prop=None, tmap=None):
                #     tmap[np.isnan(tmap)]=-9999
                #     tmap = tmap.astype(int)  # make sure the map has integer tile values
                #     print(np.unique(tmap))
                #     ntiles = len(ptid)
                #     prop_dict = {ptid[i]: prop[i] for i in range(ntiles)}
                #     prop_dict[-9999] = np.nan
                #     def vec_translate(a, my_dict):
                #         return np.vectorize(my_dict.__getitem__)(a)
                #     mapped_property = vec_translate(tmap, prop_dict)
                #     return mapped_property
                #
                #
                # gtd.mappedt_fluxes.fdir =  map_tiled_prop(gtd.tile, gtd.tiled_fluxes.fdir,  gtd.tiles_hr_map.data)
                # gtd.mappedt_fluxes.fdif =  map_tiled_prop(gtd.tile, gtd.tiled_fluxes.fdif,  gtd.tiles_hr_map.data)
                # gtd.mappedt_fluxes.frdir = map_tiled_prop(gtd.tile, gtd.tiled_fluxes.frdir, gtd.tiles_hr_map.data)
                # gtd.mappedt_fluxes.frdif = map_tiled_prop(gtd.tile, gtd.tiled_fluxes.frdif, gtd.tiles_hr_map.data)
                # gtd.mappedt_fluxes.fcoup = map_tiled_prop(gtd.tile, gtd.tiled_fluxes.fcoup, gtd.tiles_hr_map.data)

                # Instead of computing fluxes on mapped tiles, first compute flux for each tile and then map them::
                gtd.mappedtll_fluxes.fdir =  gf.map_tiled_prop(gtd.tile, gtd.tiled_fluxes.fdir,  gtd.tiles_hr_map.data)
                gtd.mappedtll_fluxes.fdif =  gf.map_tiled_prop(gtd.tile, gtd.tiled_fluxes.fdif,  gtd.tiles_hr_map.data)
                gtd.mappedtll_fluxes.frdir = gf.map_tiled_prop(gtd.tile, gtd.tiled_fluxes.frdir, gtd.tiles_hr_map.data)
                gtd.mappedtll_fluxes.frdif = gf.map_tiled_prop(gtd.tile, gtd.tiled_fluxes.frdif, gtd.tiles_hr_map.data)
                gtd.mappedtll_fluxes.fcoup = gf.map_tiled_prop(gtd.tile, gtd.tiled_fluxes.fcoup, gtd.tiles_hr_map.data)



                gtd.mappedt_fluxes.fdir = gf.latlon_to_equal_area(gtd.mappedtll_fluxes.fdir, gtd.lat, gtd.lon, outdir=outfigdir_ik_rasters, myvar='fdir2')
                gtd.mappedt_fluxes.fdif = gf.latlon_to_equal_area(gtd.mappedtll_fluxes.fdif, gtd.lat, gtd.lon, outdir=outfigdir_ik_rasters, myvar='fdif2')
                gtd.mappedt_fluxes.frdir = gf.latlon_to_equal_area(gtd.mappedtll_fluxes.frdir, gtd.lat, gtd.lon, outdir=outfigdir_ik_rasters, myvar='frdir2')
                gtd.mappedt_fluxes.frdif = gf.latlon_to_equal_area(gtd.mappedtll_fluxes.frdif, gtd.lat, gtd.lon, outdir=outfigdir_ik_rasters, myvar='frdir2')
                gtd.mappedt_fluxes.fcoup = gf.latlon_to_equal_area(gtd.mappedtll_fluxes.fcoup, gtd.lat, gtd.lon, outdir=outfigdir_ik_rasters, myvar='fcoup2')

        ###############################################################
        ### NOW HERE WE PLOT RES. FOR CURRENT DOMAIN AND INPUTKINDKIND


        outfigdir_ik_stats = os.path.join(outfigdir_id, "{}_stats_plots".format(my_input_kind))
        os.system("mkdir -p {}".format(outfigdir_ik_stats))
        file_res = os.path.join(outfigdir_ik_stats, 'tiled_metrics.nc')
        # dsres.to_netcdf(file_res)
        # outfigdir = os.path.join(outfigdir_ik_stats, 'tile_stats_plots')
        # os.system("mkdir -p {}".format(outfigdir))
        # gf.plot_tiled_stats(dsres, outfigdir=outfigdir, var='terrains')
        # for mymodel in models:
        #     gf.plot_tiled_stats(dsres, outfigdir=outfigdir, var='fluxes', model=mymodel)

        # PLOT 4 MAPS / 4 HISTS
        outfigdir_mapped = os.path.join(outfigdir_ik_stats, 'tile_mapped_plots')
        os.system("mkdir -p {}".format(outfigdir_mapped))

        # plot_4_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, plot_differences=True)
        # plot_4_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, plot_differences=False)
        # plot_4_tiled_hist(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped)
        # plot_tiled_scatter_plots(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, scattering_aveblock = 6, do_aveblock=False)

        gf.plot_2tiles_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped)

        gf.plot_4_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, plot_differences=True)
        gf.plot_4_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, plot_differences=False)
        gf.plot_4_tiled_hist(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped)
        gf.plot_tiled_scatter_plots(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, scattering_aveblock = 6, do_aveblock=False, do_hexbin=False)
        gf.plot_tiled_scatter_plots(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, scattering_aveblock = 6, do_aveblock=False, do_hexbin=True)
        gf.plot_tiled_histogram_plots(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, scattering_aveblock=6, do_aveblock=False)
        # plot_tiled_histogram_plots(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, scattering_aveblock=6, do_aveblock=False)

# np.min(RES_ALL[5].mappedt_fluxes.frdir)
# np.min(RES_ALL[5].mappedtll_fluxes.frdir)


# np.min( RES_ALL[5].mappedt_fluxes.frdir )
# np.min( RES_ALL[5].mappedtll_fluxes.frdir )
# np.min( RES_ALL[5].tiled_fluxes.frdir )


#
file_res = os.path.join(conv_outfigdir, 'tiled_metrics.nc')
dsres.to_netcdf(file_res)
dsres = xr.open_dataset(file_res)
# outfigdir = os.path.join(outfigdir, 'tile_stats_plots')
# os.system("mkdir -p {}".format(outfigdir))
# gf.plot_tiled_stats(dsres, outfigdir=outfigdir, var='terrains', myinputkind=my_input_kind )
# for mymodel in models:
#     gf.plot_tiled_stats(dsres, outfigdir=outfigdir, var='fluxes', model=mymodel, myinputkind=my_input_kind )


mystats = ['stdv', 'skew', 'kurt']
for mik in inputkinds:
    gf.plot_tiled_stats(dsres, outfigdir=conv_outfigdir, var='terrains', myinputkind=mik, mystats=tuple(mystats))
    for mymodel in models:
        gf.plot_tiled_stats(dsres, outfigdir=conv_outfigdir, var='fluxes', model=mymodel, myinputkind=mik, mystats=tuple(mystats))

# plot 2 different tiling schemes in the same plot
inputkinds = ['k5n1pV','kVn1p5']
mik1 = 'k5n1pV'
mik2 = 'kVn1p5'
mystats = ['stdv', 'skew', 'kurt']
for mik in inputkinds:
    gf.plot_tiled_stats(dsres, outfigdir=conv_outfigdir, var='terrains', myinputkind=mik1, myinputkind2 = mik2, mystats=tuple(mystats))
    for mymodel in models:
        gf.plot_tiled_stats(dsres, outfigdir=conv_outfigdir, var='fluxes', model=mymodel, myinputkind=mik1, myinputkind2 = mik2, mystats=tuple(mystats))


# for mik in inputkinds:
#     plot_tiled_stats(dsres, outfigdir=outfigdir, var='terrains', myinputkind=mik )
#     for mymodel in models:
#         plot_tiled_stats(dsres, outfigdir=outfigdir, var='fluxes', model=mymodel, myinputkind=mik )

# XA0 = getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, 'fdif')
# # print(XA0)
# # XA = np.ravel(dem.crop_and_average(XA0, buffer=1, average=True, aveblock=6))
# YA0 = getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, 'fdif').data
# print(YA0.shape)
# # XA = np.ravel(dem.crop_and_average(XA, buffer=1, average=True, aveblock=6))
#
#
# plt.figure()
# plt.imshow(XA0)
# plt.savefig(os.path.join(outfigdir_mapped, 'prova_XA.png'))
# plt.show()
# plt.figure()
# plt.imshow(YA0)
# plt.savefig(os.path.join(outfigdir_mapped, 'prova_YA.png'))
# plt.show()

# np.mean(gtd.tiled_fluxes.fdir)
# np.mean(gtd.hires_fluxes.fdir)

# print( np.mean(RES_ALL[2].tiled_fluxes.fdir))
# print( np.mean(RES_ALL[2].hires_fluxes.fdir))
# print( np.mean(RES_ALL[5].tiled_fluxes.fdir))
# print( np.mean(RES_ALL[50].tiled_fluxes.fdir))

# y_highres = dsres.sel(domains=mydom, labels=myvar, stats=mystat, models=model).values
# y_tiled = datp.sel(domains=mydom, labels=myvar, stats=mystat, models=model).values

# dsres['hires_fluxes'].sel(stats='mean', labels='fdir')
# dsres['tiled_fluxes'].sel(stats='mean', labels='fdir')

### develop new plot
#
# # import matplotlib
# # matplotlib.use("qt4agg")
# # matplotlib.use('MacOSX')
#
# # matplotlib.use('TkAgg')
# # matplotlib.use('GTK4Agg')
#
# # RES_ALL[5].hires_fluxes
#
# from pylab import cm
# myvar = 'tiles_hr_map'
# # d0 = RES_ALL[NPV_2_PLOT[0]]
# d0 = RES_ALL[1]
# HRMA = np.array(getattr(d0, "{}".format(myvar)).data)
# HRMA2 = HRMA.astype(int)
# HRMA[HRMA < -999] = np.nan
# HRMA2[HRMA2 < -999] = 0 # only for computing colorbar
# ntiles = len(np.unique(HRMA2))
# cmap = cm.get_cmap('Pastel1', ntiles)    # 11 discrete colors
#
#
#
# plt.figure()
# mat = plt.pcolormesh(d0.LON, d0.LAT, HRMA, cmap=cmap, vmin=np.nanmin(HRMA)-0.5, vmax=np.nanmax(HRMA)+0.5)
# # cax = plt.colorbar(mat, ticks=np.unique(HRMA2))
# plt.colorbar()
# plt.savefig("1.png")
# plt.close()
# #
# # d0.tiles_hr_map
# #
# # d0.tiles_hr_map.data
# #
# #
#
# def plot_6_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=None):
#     # gf.plot_6_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, plot_differences=True)
#
#     # allvars = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup', 'sian', 'tcf', 'svf']
#     # plot tiled configuration together with one variable, for different ntiles
#     allvars = ['tcf']
#     d0 = RES_ALL[NPV_2_PLOT[0]]
#     d1 = RES_ALL[NPV_2_PLOT[1]]
#     # plot_differences = True
#
#     # do this when using block averaging on high res data
#     # aveblock_data = np.flipud(getattr(d0.hires_fluxes, "{}".format('sian')).data)
#     aveblock_data = getattr(d0.hires_fluxes, "{}".format('sian')).data
#     # aveblock_lat = np.linspace(np.min(d0.LATea), np.max(d0.LATea), np.shape(aveblock_data)[0])
#     aveblock_lat = np.linspace(np.max(d0.LATea), np.min(d0.LATea), np.shape(aveblock_data)[0])
#     aveblock_lon = np.linspace(np.min(d0.LONea), np.max(d0.LONea), np.shape(aveblock_data)[1])
#     AVELON, AVELAT = np.meshgrid(aveblock_lon, aveblock_lat)
#
#
#
#     # iv = 0
#     # myvar = allvars[0]
#
#     for iv, myvar in enumerate(allvars):
#         # bnd = BND[iv]
#         up_bnd0 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data, 0.95)
#         do_bnd0 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data, 0.05)
#         up_bnd1 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data, 0.95)
#         do_bnd1 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data, 0.05)
#
#         # up_bnd0 = np.max(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data)
#         # do_bnd0 = np.max(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data)
#         # up_bnd1 = np.max(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data)
#         # do_bnd1 = np.max(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data)
#         up_bnd = max(up_bnd0, up_bnd1)
#         do_bnd = max(do_bnd0, do_bnd1)
#         # print('bounds for {} :{:.3f} - {:.3f}'.format(myvar, do_bnd, up_bnd))
#         fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
#         # cm0 = axes[0,0].pcolormesh(d0.LONea, d0.LATea, np.flipud(getattr(d0.hires_fluxes, "{}".format(myvar)).data), cmap='jet', vmin = do_bnd, vmax = up_bnd)
#         # TILES0 = np.array(getattr(d0, "{}".format('tiles_ea_map')).data)
#         TILES0 = np.array(getattr(d0, "{}".format('tiles_hr_map')).data)
#         TILES1 = np.array(getattr(d1, "{}".format('tiles_hr_map')).data)
#         VAR0 = np.array(getattr(d0, "{}".format(myvar)).data)
#         VAR1 = np.array(getattr(d1, "{}".format(myvar)).data)
#
#         aveblock_lat_tiles = np.linspace(np.max(d0.LAT), np.min(d0.LAT), np.shape(TILES0)[0])
#         aveblock_lon_tiles = np.linspace(np.min(d0.LON), np.max(d0.LONea), np.shape(TILES0)[1])
#         AVTLON, AVTLAT = np.meshgrid(aveblock_lon_tiles, aveblock_lat_tiles)
#         # AVTLON, AVTLAT = np.meshgrid(d0.LON, d0.LAT)
#
#         TILES0_INT = TILES0.astype(int)
#         TILES0_INT[TILES0_INT<-999] = 0 # for colorbar purposes only
#         TILES0[TILES0<-999] = np.nan # values to plot
#         TILES1_INT = TILES1.astype(int)
#         TILES1_INT[TILES1_INT<-999] = 0 # for colorbar purposes only
#         TILES1[TILES1<-999] = np.nan # values to plot
#         cmap_tiles_0 = cm.get_cmap('Pastel1', len(np.unique(TILES0_INT)))
#         cmap_tiles_1 = cm.get_cmap('Pastel1', len(np.unique(TILES1_INT)))
#
#         # mat = plt.pcolormesh(d0.LON, d0.LAT, HRMA, cmap=cmap, vmin=np.nanmin(HRMA) - 0.5, vmax=np.nanmax(HRMA) + 0.5)
#
#         cm0t = axes[0, 0].pcolormesh(AVTLON, AVTLAT, TILES0,
#                 cmap=cmap_tiles_0, vmin=np.nanmin(TILES0) - 0.5, vmax=np.nanmax(TILES0) + 0.5)
#
#         cm1t = axes[0, 1].pcolormesh(AVTLON, AVTLAT, TILES1,
#                 cmap=cmap_tiles_1, vmin=np.nanmin(TILES1) - 0.5, vmax=np.nanmax(TILES1) + 0.5)
#
#         data1 = getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data
#         data2 = getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data
#             # data3 = HRMA - getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data
#         cm1 = axes[1, 0].pcolormesh(d0.LONea, d0.LATea, data1, cmap='bwr', vmin=0, vmax=up_bnd)
#         cm2 = axes[1, 1].pcolormesh(d0.LONea, d0.LATea, data2, cmap='bwr', vmin=0, vmax=up_bnd)
#             # cm3 = axes[1, 1].pcolormesh(d0.LONea, d0.LATea, data3, cmap='bwr', vmin=-2, vmax=2)
#         savename_4plots = 'mapped_2tiles'
#
#         fig.colorbar(cm0t, ax=axes[0,0])
#         fig.colorbar(cm1t, ax=axes[0,1])
#         fig.colorbar(cm1, ax=axes[1,0], extend='max')
#         fig.colorbar(cm2, ax=axes[1,1], extend='max')
#
#
#         axes[0, 0].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[0]].ntiles_only_soil))
#         axes[0, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[1]].ntiles_only_soil))
#         axes[1, 0].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[0]].ntiles_only_soil))
#         axes[1, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[1]].ntiles_only_soil))
#         # axes[1, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[2]].ntiles_only_soil))
#
#         axes[0, 0].set_ylabel('Latitude')
#         axes[0, 0].set_xlabel('Longitude')
#         axes[1, 0].set_ylabel('Latitude')
#         axes[1, 0].set_xlabel('Longitude')
#
#         axes[0, 1].set_ylabel('Latitude')
#         axes[0, 1].set_xlabel('Longitude')
#         axes[1, 1].set_ylabel('Latitude')
#         axes[1, 1].set_xlabel('Longitude')
#
#
#         countp = 0
#         for j in [0, 1]:
#             for i in [0, 1]:
#                 axes[j, i].text(-0.1, 1.1, string.ascii_uppercase[countp],
#                                 transform=axes[j, i].transAxes,
#                                 size=20, weight='bold')
#                 countp += 1
#         plt.tight_layout()
#         plt.savefig(os.path.join(outfigdir, '{}_{}.png'.format(savename_4plots, myvar)))
#         # plt.savefig('1.png')
#         plt.close()
#     return
