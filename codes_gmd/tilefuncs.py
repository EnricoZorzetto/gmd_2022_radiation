import os
import string
import numpy as np

from geospatialtools import gdal_tools
# import tensorflow as tf
import demfuncs as dem
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg') # dyn show plots

dem.matplotlib_update_settings()

# ------------------------------------------------------------------------------
# Functions needed for analyzing and plotting the results
# of nate's GFDL preprocessing module
# ------------------------------------------------------------------------------



def dprint(dict):
    for ki in list(dict.keys()):
        v = dict[ki]
        print('{}:: mean = {:.4f}, stdv = {:.4f}, min = {:.4f}, max = {:.4f}'.format(
            ki, np.mean(v), np.std(v), np.min(v), np.max(v)))


def skew(sample):

    # sample1 = sample[~np.isnan(sample)]
    sigma = np.nanstd(sample)
    mu = np.nanmean(sample)
    xn = (sample - mu)/sigma
    myskew = np.nanmean( xn**3)
    return myskew


def kurt(sample):
    # sample1 = sample[~np.isnan(sample)]
    sigma = np.nanstd(sample)
    mu = np.nanmean(sample)
    xn = (sample - mu) / sigma
    mykurt = np.nanmean( xn ** 4)
    return mykurt


def comp_4_stats(sample, weights = None):
    res = {}
    if weights is None:
        res['mean'] = np.nanmean(sample)
        res['stdv'] = np.nanstd(sample)
        res['skew'] = skew(sample)
        res['kurt'] = kurt(sample)
    else:
        # do it only for vectors (=> list of tiles)
        assert np.size(weights) == np.size(sample)
        assert np.ndim(weights) == 1
        res['mean'] = weighted_stats(sample, weights, k=1)
        res['stdv'] = weighted_stats(sample, weights, k=2)
        res['skew'] = weighted_stats(sample, weights, k=3)
        res['kurt'] = weighted_stats(sample, weights, k=4)
    return res


def weighted_stats(x, f, k=1):
    # x -> samples, f -> weights, k -> order
    ### TEST::
    # A = np.array([-9999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 7, 7, 12, 12, 121])
    # X = np.unique(A)
    # F = np.array([np.size(A[A == x]) / np.size(A) for x in X])
    # res = til.comp_4_stats(X, weights=F)
    # print(res['mean'], np.mean(A))
    # print(res['stdv'], np.std(A))
    # print(res['skew'], til.skew(A))
    # print(res['kurt'], til.kurt(A))
    if k == 1:
        res = np.sum(x*f)
    elif k==2:
        mu = weighted_stats(x, f, k=1)
        res = np.sqrt( np.sum(f*(x-mu)**2) )
    elif k==3:
        mu = weighted_stats(x, f, k=1)
        sigma = weighted_stats(x, f, k=2)
        res = np.sum(f*(x-mu)**3)/sigma**3
    elif k==4:
        mu = weighted_stats(x, f, k=1)
        sigma = weighted_stats(x, f, k=2)
        res = np.sum(f*(x-mu)**4)/sigma**4
    else:
        print("k={}".format(k))
        # raise Exception("order k must be in [1,2,3,4]")
    return res


def maphrus(hrus, hruprops,
            prop='hillslope_aspect'):
    """ map a terrain properties based on the map of tiles
      and the avergae property for each tile"""
    ny, nx = hrus.shape
    print(ny, nx)
    map_prop = np.zeros((ny, nx) ) *np.nan
    for i in range(ny):
        for j in range(nx):
            my_hru_index = hrus[i, j]
            if not np.isnan(my_hru_index) and my_hru_index > -1:
                my_hru_index = hrus[i ,j] - 1
                my_value = hruprops[prop][my_hru_index]
                map_prop[i ,j] = my_value
    return map_prop




def read_tile_properties(datadir=None, do_averaging=True, aveblock=110,
                         cosz = None, phi = None, adir=None,
                         modeldir=None):
    """-------------------------------------------------------------------------
    Read the output of Nate's GFDL: preprocessing stored in folder datadir
    read the maps of terrain quantities of interest
    and the tile-by-tile average values of these quantities

    cosz, phi -> Zenith and azimuth angles needed to compute the
    tile-by-tile solar incidence angle

    modeldir -> folder with the models used to compute rad fluxes differences
    -------------------------------------------------------------------------"""
    landfolder = os.path.join(datadir, 'land', 'tile:1,is:1,js:1')

    # read tiles and their areal-average properties
    with open(os.path.join(landfolder, 'hrus.pck'), "rb") as hru_file:
        hrus = pickle.load(hru_file)
        # print("shape of hrus = {}".format(hrus.shape))
        # hrus = np.flipud(hrus)
    with open(os.path.join(landfolder, 'hru_properties.pck'), "rb") as hru_file2:
        hruprops = pickle.load(hru_file2)

    # hillsfile = os.path.join(landfolder, 'hillslopes_ea.tif')
    # hillsfile0 = os.path.join(landfolder, 'ti_ea.tif')
    hillsfile = os.path.join(landfolder, 'tiles.tif')
    # hills0 = gdal_tools.read_data(hillsfile0).data
    hills = gdal_tools.read_data(hillsfile).data

    # with open(os.path.join(landfolder, 'hillslope_properties.pck')) as hill_file:
    #     hillprops = pickle.load(hill_file)

    # matplotlib.use('Qt5Agg') # dyn show plots
    # plt.figure()
    # plt.imshow(hills0)
    # plt.show()
    # hills[hills<0] = -1
    # plt.figure()
    # plt.imshow(hills)
    # plt.colorbar()
    # plt.show()
    # print('hills plot')


    # read the DEM or some other dataset in the equal area projection
    elevfile_latlon = os.path.join(landfolder, 'dem_latlon.tif')
    elevfile = os.path.join(landfolder, 'demns_ea.tif')
    # cossfile = os.path.join(landfolder, 'coss_ea.tif')
    sinscosafile = os.path.join(landfolder, 'sinscosa_ea.tif')
    sinssinafile = os.path.join(landfolder, 'sinssina_ea.tif')
    tcffile = os.path.join(landfolder, 'tcf_ea.tif')
    svffile = os.path.join(landfolder, 'svf_ea.tif')
    radavelevfile = os.path.join(landfolder, 'radavelev_ea.tif')
    radstelevfile = os.path.join(landfolder, 'radstelev_ea.tif')


    # RD = gdal_tools.read_data(elevfile)

    elev_map_latlon = gdal_tools.read_data(elevfile_latlon)
    minlat = elev_map_latlon.miny
    maxlat = elev_map_latlon.maxy
    minlon = elev_map_latlon.minx
    maxlon = elev_map_latlon.maxx

    elev_map_raster = gdal_tools.read_data(elevfile)
    elev_map = elev_map_raster.data
    tcf_map = gdal_tools.read_data(tcffile).data
    svf_map = gdal_tools.read_data(svffile).data
    # coss_map = gdal_tools.read_data(cossfile).data
    sinssina_map = gdal_tools.read_data(sinssinafile).data
    sinscosa_map = gdal_tools.read_data(sinscosafile).data
    radstelev_map = gdal_tools.read_data(radstelevfile).data
    radavelev_map = gdal_tools.read_data(radavelevfile).data

    # nx, ny = elev_map.shape
    # print(elev_map.shape)
    # print(radavelev_map.shape)

    nx = elev_map_raster.nx
    ny = elev_map_raster.ny
    ylat_ea = np.flipud( np.linspace(minlat, maxlat, ny) )
    xlon_ea = np.linspace(minlon, maxlon, nx)

    nx_latlon = elev_map_latlon.nx
    ny_latlon = elev_map_latlon.ny
    ylat_latlon = np.flipud( np.linspace(minlat, maxlat, ny_latlon) )
    xlon_latlon = np.linspace(minlon, maxlon, nx_latlon)


    # if do_averaging:
    #
    #     avblock = aveblock
    #
    #     # block average terrain maps
    #     elev_map =      dem.crop_and_average(elev_map,         average=True, aveblock=avblock)
    #     sinscosa_map =  dem.crop_and_average(sinscosa_map,     average=True, aveblock=avblock)
    #     sinssina_map =  dem.crop_and_average(sinssina_map,     average=True, aveblock=avblock)
    #     tcf_map =       dem.crop_and_average(tcf_map,          average=True, aveblock=avblock)
    #     svf_map =       dem.crop_and_average(svf_map,          average=True, aveblock=avblock)
    #     radstelev_map = dem.crop_and_average(radstelev_map,    average=True, aveblock=avblock)
    #     radavelev_map = dem.crop_and_average(radavelev_map,    average=True, aveblock=avblock)



    # create coordinates for original elevation map (equal area projection)::
    elev = gdal_tools.read_data(elevfile)
    x_ea = np.linspace(elev.minx, elev.maxx, elev.nx)
    y_ea = np.linspace(elev.miny, elev.maxy, elev.ny)

    ntiles = len(hruprops['hillslope_aspect'])
    print('number of HRUs = {}'.format(ntiles))
    print(np.unique(hrus))
    print(np.shape(hrus))

    hrus_plot = hrus.copy().astype(float)
    hrus_plot[hrus_plot < 0] = np.nan

    sinz = (1 - cosz ** 2) ** (0.5)

    def compute_sia(cosz=None, sinz=sinz, phi=None, SS=None, SC=None):
        sia = cosz + sinz * ( np.sin(phi)*SS + np.cos(phi)*SC)
        return sia

    tile_sia = compute_sia(cosz=cosz, sinz=sinz, phi=phi,
                        SS=hruprops['hillslope_sinssina'],
                        SC=hruprops['hillslope_sinscosa'])

    sia_map = compute_sia(cosz=cosz, sinz=sinz, phi=phi,
                       SS=sinssina_map, SC=sinscosa_map)

    hruprops['hillslope_sia'] = tile_sia

    # return a dictionary with tile data
    res = {}
    res['ntiles'] = ntiles
    res['cosz'] = cosz
    res['phi'] = phi

    # coordinates and bounds

    res['y_ea'] = y_ea # coordinates in [km], equal area map
    res['x_ea'] = x_ea # coordinates in [km], equal area map
    res['ylat_ea'] = ylat_ea # coorinates in [lat, lon], equal area map
    res['xlon_ea'] = xlon_ea # coorinates in [lat, lon], equal area map
    res['ylat_latlon'] = ylat_latlon # latlon coordinates, latlon map
    res['xlon_latlon'] = xlon_latlon # latlon coordinates, latlon map
    res['minlat'] = minlat # domain boundaries, lat-lon
    res['maxlat'] = maxlat # domain boundaries, lat-lon
    res['minlon'] = minlon # domain boundaries, lat-lon
    res['maxlon'] = maxlon # domain boundaries, lat-lon

    # average value for each tile
    res['tile_ele'] = hruprops['hillslope_radavelev']
    res['tile_sde'] = hruprops['hillslope_radstelev']
    res['tile_svf'] = hruprops['hillslope_svf']
    res['tile_tcf'] = hruprops['hillslope_tcf']
    res['tile_sia'] = hruprops['hillslope_sia']

    # average value for each tile, mapped over domain
    res['mappedtile_ele'] = maphrus(hrus, hruprops, 'hillslope_radavelev')
    res['mappedtile_sde'] = maphrus(hrus, hruprops, 'hillslope_radstelev')
    res['mappedtile_svf'] = maphrus(hrus, hruprops, 'hillslope_svf')
    res['mappedtile_tcf'] = maphrus(hrus, hruprops, 'hillslope_tcf')
    res['mappedtile_sia'] = maphrus(hrus, hruprops, 'hillslope_sia')

    # if do_averaging:
    #     res['mappedtile_ele'] = dem.crop_and_average( res['mappedtile_ele'], average=True, aveblock=aveblock)
    #     res['mappedtile_sde'] = dem.crop_and_average( res['mappedtile_sde'], average=True, aveblock=aveblock)
    #     res['mappedtile_svf'] = dem.crop_and_average( res['mappedtile_svf'], average=True, aveblock=aveblock)
    #     res['mappedtile_tcf'] = dem.crop_and_average( res['mappedtile_tcf'], average=True, aveblock=aveblock)
    #     res['mappedtile_sia'] = dem.crop_and_average( res['mappedtile_sia'], average=True, aveblock=aveblock)

    res['mappedtile_hrus'] = hrus
    res['mappedtile_hills'] = hills


    # values of original high res (or ave) maps
    res['map_ele'] = radavelev_map
    res['map_sde'] = radstelev_map
    res['map_svf'] = svf_map
    res['map_tcf'] = tcf_map
    res['map_sia'] = sia_map

    res['map_sinssina'] = sinssina_map
    res['map_sinscosa'] = sinscosa_map







    # LOAD MLR MODEL AND USE IT FOR PREDICTIONS
    FLUX_TERMS = ['fdir', 'frdirn', 'fdif', 'frdifn', 'fcoupn']
    for flux_term in FLUX_TERMS:

        # APPLY THE MODEL TO MAKE PREDICTIONS TILE-BY-TILE:
        dfhru = pd.DataFrame({
            # 'ele':hruprops['hillslope_radavelev'],
            # 'sde':hruprops['hillslope_radstelev'],
            'tcfn': hruprops['hillslope_tcf'],
            'svfn': hruprops['hillslope_svf'],
            'sian': tile_sia
        })

        # MAKE "TRUE" PREDICTIONS BASED ON HIGH-RESOLUTION ORIGINAL TERRAIN DATA::
        # using the equal area projection map, averaged if needed
        dfhrt = pd.DataFrame({
            # 'ele': radavelev_map.flatten(),
            # 'sde': radstelev_map.flatten(),
            'tcfn': tcf_map.flatten(),
            'svfn': svf_map.flatten(),
            'sian': sia_map.flatten()
        })

        # EZDEV: keep SIAN, changed order or predictors wrt version 1
        # if flux_term in ['frdirn', 'frdifn'] or (cosz > 0.99 and flux_term == 'fdir'):
        #     _ = dfhrt.pop('sian')
        #     _ = dfhru.pop('sian')

        mlr_savedir = modeldir
        mlr_savename = os.path.join(mlr_savedir,
                   'mlr_model_{}_cosz_{}_adir_{}'.format(flux_term, cosz, adir))
        with open(os.path.join(mlr_savedir,
                   '{}.pickle'.format(mlr_savename)), 'rb') as pklf:
            mlr_model = pickle.load(pklf)

        print('model char::')
        print(mlr_model.n_features_in_)
        print(dfhru.shape)
        print(dfhrt.shape)
        print('end model diagnostics')

        # compute fluxes
        # hru_pred_tiles = mlr_model.predict(dfhru).flatten()
        tile_pred = mlr_model.predict(dfhru)
        map_pred_1d = mlr_model.predict(dfhrt)

        # hruprops[flux_term] = hru_pred_tiles

        # plt.figure()
        # plt.imshow(map_pred_hru)
        # plt.colorbar()
        # plt.show()

        # MAP PREDICTIONS:
        # map_pred_hru = maphrus(hrus, hruprops, prop=flux_term)

        # average tile-by-tile predictions at same scale used for terrain vars
        # map_pred_ave_hru = dem.crop_and_average(tile_pred,
        #                         average=True, aveblock=aveblock)

        map_pred = map_pred_1d.reshape( np.shape(elev_map) )
        # hru_pred = map_pred_ave_hru.flatten()

        # save results to output for current flux component
        res['tile_{}'.format(flux_term)] = tile_pred
        res['map_{}'.format(flux_term)] = map_pred

        hruprops['hillslope_{}'.format(flux_term)] = tile_pred

        tile_pred_mapped = maphrus(hrus, hruprops,
                                    'hillslope_{}'.format(flux_term))

        if do_averaging:
            tile_pred_mapped_averaged = dem.crop_and_average(tile_pred_mapped,
                                average=True, aveblock=aveblock)
        else:
            tile_pred_mapped_averaged = tile_pred_mapped



        # res['mappedtile_{}'.format(flux_term)] = tile_pred_mapped
        res['mappedtile_{}'.format(flux_term)] = tile_pred_mapped_averaged


    return res





