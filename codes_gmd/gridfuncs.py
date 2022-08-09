
import string
import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import h5py
import matplotlib.pyplot as plt
import demfuncs as dem
from geospatialtools import gdal_tools
import gdal_write_raster as rster
from copy import deepcopy
import matplotlib as mpl
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

from pylab import cm



def rescale_preserving_minval(X, Ybar=1.0, lowerbound = -1.0, weights=None):

    Xmin = np.min(X)
    if weights is not None:
        Xbar = np.sum(weights * X)
    else:
        Xbar = np.mean(X)
    Ymin = max(Ybar - (Xbar - Xmin), lowerbound)
    if (Xbar <= Xmin) or (Ybar <= Ymin):
        Y = np.ones(np.shape(X))*Xbar
    else:
        Y = Ybar + (Ybar-Ymin)*(X-Xbar)/(Xbar-Xmin)
    return Y

def latlon_to_equal_area(my_array, lat, lon, outdir=None, myvar = 'fdir'):
    # must provide a directory (outdir) where raster files will be stored
    # requires map_tiles_to_map already run
    # lat = self.yy
    # lon = self.xx
    # vars2remap = ['tcf', 'svf', 'ss', 'sc']
    # vars2remap = {'tcf':None, 'svf':None, 'ss':None, 'sc':None, 'elen':None}
    # vars2remap ={var:None for var in vars}
    os.system("mkdir -p {}".format(os.path.join(outdir, 'tile_mapped_rasters')))
    # for myvar in vars:
    infile_ll = os.path.join(outdir, 'tile_mapped_rasters', '{}_map_ll.tif'.format(myvar))
    outfile_ea = os.path.join(outdir, 'tile_mapped_rasters', '{}_map_ea.tif'.format(myvar))
    logfile_ea = os.path.join(outdir, 'tile_mapped_rasters', '{}_map_ea.log'.format(myvar))
    # my_array = getattr(self, "{}_mapped".format(myvar))

    # rster.write_raster_WGS84_ezdev(infile_ll, my_array, self.lat, self.lon, nodata=np.nan)
    rster.write_raster_WGS84_ezdev(infile_ll, my_array, lat, lon, nodata=-9999)
    rster.write_ea(infile_ll, outfile_ea, logfile_ea, eares=90.0, interp='average')
    mydata_ea = gdal_tools.read_data(outfile_ea)
    # mydata_ea

    mydata_ea.data[mydata_ea.data < -9000.0] = np.nan

    return mydata_ea.data


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
        # print("k={}".format(k))
        raise Exception("order k must be in [1,2,3,4]")
    return res


def init_metric_tiles(domains, npvalues, inputkinds, models, labels, terrains, stats,
                      cosz=0.3, azi=0.0, adir=0.3,
                      prefit_model_aveblock=None, prefit_model_buffer=None,
                      prefit_model_adir=None, prefit_model_domain=None
                      ):
    # initialize a dataset to store model result
    # note: cosz and adir (floats) must be indexed using their integer index
    nterrains = len(terrains)
    nlabels = len(labels)
    ndomains = len(domains)
    ninputkinds = len(inputkinds)
    nnpvalues = len(npvalues)
    nmodels = len(models)
    nstats = len(stats)

    modeldata = xr.Dataset(
        {
            "tiled_fluxes": (
                ("labels", "domains", "inputkinds", "npvalues", "models", "stats"),
                np.zeros((nlabels, ndomains, ninputkinds,
                          nnpvalues, nmodels, nstats),
                         dtype=np.float32)),
            "hires_fluxes": (
                ("labels", "domains", "models", "stats"),
                np.zeros((nlabels, ndomains,
                          nmodels, nstats),
                         dtype=np.float32)),

            "tiled_terrains": (
                ("terrains", "domains", "inputkinds", "npvalues", "stats"),
                np.zeros((nterrains, ndomains, ninputkinds,
                          nnpvalues, nstats),
                         dtype=np.float32)),
            "hires_terrains": (
                ("terrains", "domains", "stats"),
                np.zeros((nterrains, ndomains,
                          nstats),
                         dtype=np.float32)),
        },
        coords={
            "terrains": terrains,
            "inputkinds": inputkinds,
            "npvalues": npvalues,
            "models": models,
            "labels": labels,
            "stats": stats,
            "domains": domains,
        },
        attrs={
            "prefit_model_aveblock": prefit_model_aveblock,
            "prefit_model_buffer": prefit_model_buffer,
            "prefit_model_domain": prefit_model_domain,
            "prefit_model_adir": prefit_model_adir,
            "cosz": cosz, # cosz value used to evaluate the models
            "azi": azi, # azimuth angle used to evaluate the models
            "adir": adir  # albedo used to train and evaluate the models
        }
    )
    return modeldata


def npk_from_string(my_input_kind='kVn1p5', myx=1):
    myk = my_input_kind[1]
    myn = my_input_kind[3]
    myp = my_input_kind[5]
    # outfigdir_ip = os.path.join(outfigdir_ik, "{}".format(my_input_kind))
    # os.system("mkdir -p {}".format(outfigdir_ip))
    if myk == 'V':
        myk = myx
        varyingpar = 'k'
    elif myn == 'V':
        myn = myx
        varyingpar = 'n'
    elif myp == 'V':
        myp = myx
        varyingpar = 'p'
    else:
        raise Exception("Errr: must specify a valid <my_input_kind> value!")
    return myk, myn, myp, varyingpar


def map_tiled_prop(ptid=None, prop=None, tmap=None):
    """
    : given a list of tiles (ptid) and corresponding properties (prop)
    : map them on a map (tmap) of tile values
    :return: array with mapped property

    # Example of usage::

    tiles_pred.tcf_mapped = gf.map_tiled_prop(gtd.tile, tiles_pred.tcf, gtd.tiles_hr_map)
    """
    tmap[np.isnan(tmap)] = -9999
    tmap = tmap.astype(int) # make sure the map has integer tile values
    # ptid = gtd.tile
    # myT = gtd.svf
    # ntiles = gtd.ntiles
    # arr2 = gtd.tiles_hr_map.data.astype(int)
    ntiles = len(ptid)
    prop_dict = {ptid[i]:prop[i] for i in range(ntiles)}
    prop_dict[-9999] = np.nan
    def vec_translate(a, my_dict):
        return np.vectorize(my_dict.__getitem__)(a)
    mapped_property = vec_translate(tmap, prop_dict)
    # mapped_property = np.flipud( mapped_property) # do it only for plotting
    return mapped_property


class ucla_terrain:
    def __init__(self, domain="EastAlps"):
        if domain == 'EastAlps':
            self.svf = 0.981315
            self.tcf = 0.812428E-01
            self.sc = 0.619293E-02 # WEST FACING
            self.ss =-0.127051E-01 # SOUTH FACING
        elif domain == 'Nepal':
            self.svf =0.979923
            self.tcf = 0.973125E-01
            self.sc =  - 0.973074E-02 #EAST FACING
            self.ss =- 0.227080E-01   #SOUTH FACING
        elif domain == 'Peru':
            self.svf = 0.984905
            self.tcf = 0.727807E-01
            self.sc = - 0.289956E-02 # EAST FACING
            self.ss = 0.604277E-02 # NORTH FACING
        elif domain == 'FrenchAlps':
            self.svf = 0.984406
            self.tcf = 0.667791E-01
            self.sc = - 0.990379E-02 # EAST FACING
            self.ss = 0.875322E-03 # NORTH FACING


class prediction:

    def __init__(self, svf=None, tcf=None, ss=None, sc=None, elen=None,
                 type = 'LEE', cosz = None, azi = None, albedo = None, sian=None,
                 modeldir = None, prefit_models_adir=None,
                 normalize_by_grid_ave = False, normalize_grid_ave_modeldir = None,
                 specific_predictors = None,
                 normalize_fracs = None):
        # these can be either scalars, 1D - arrays (df-like) or 2D arrays (map-like)
        self.svf = svf
        self.tcf = tcf
        self.ss = ss
        self.sc = sc
        self.elen = elen
        self.ndims = np.ndim(ss)
        self.cosz = cosz
        self.azi = azi
        self.albedo = albedo
        if sian is None:
            # self.sian = cosz + np.sqrt(1 - cosz ** 2) * (np.cos(azi) * self.sc +
            #                                              np.sin(azi) * self.ss)
            # Normalize by cosz
            self.sian = 1.0 + np.sqrt(1 - cosz ** 2)/cosz * (np.cos(azi) * self.sc +
                                                         np.sin(azi) * self.ss)
        else:
            self.sian=sian
        # mydf = {'tcfn':self.tcf, 'svfn':self.svf,  'sian':self.sian}
        # mydf = {'tcfn':self.tcf, 'svfn':self.svf,  'sian':self.sian, 'elen':self.elen}

        # TODO: pass list of keys here to make sure we match changes in param
        # mydf = {'sian':self.sian, 'tcfn':self.tcf, 'svfn':self.svf}
        mydf = {'elen':self.elen,  'sian':self.sian,
                'tcfn':self.tcf, 'svfn':self.svf # use either depending on the dataset used
                # 'tcf0':self.tcf, 'svf0':self.svf # use either depending on the dataset used
                } # EZDEV2

        # if type == 'NON':
        #     self.fdir =  np.exp(4*self.sian)
        #     self.fdif =  5.0*self.sian; self.fdif[ self.sian < 0.6] = -1.0
        #     self.frdir = 5.0*self.sian; self.frdir[self.sian < 0.6] = -1.0
        #     self.frdif = 5.0*self.sian; self.frdif[self.sian < 0.6] = -1.0
        #     self.fcoup = 5.0*self.sian; self.frdif[self.sian < 0.6] = -1.0
        # else:
        # print(specific_predictors)
        # print(specific_predictors['fdir'])
        mydf_fdir = {x:y for x,y in zip(mydf.keys(), mydf.values()) if x in specific_predictors['fdir']}
        self.fdir = dem.model_predict_interp(mydf_fdir, label='fdir', cosz=cosz, albedo=albedo,
                            model=type, modeldir=modeldir, prefit_models_adir=prefit_models_adir)
        mydf_fdif = {x:y for x,y in zip(mydf.keys(), mydf.values()) if x in specific_predictors['fdif']}
        self.fdif = dem.model_predict_interp(mydf_fdif, label='fdif', cosz=cosz, albedo=albedo,
                            model=type, modeldir=modeldir, prefit_models_adir=prefit_models_adir)
        mydf_frdir = {x:y for x,y in zip(mydf.keys(), mydf.values()) if x in specific_predictors['frdir']}
        self.frdir = dem.model_predict_interp(mydf_frdir, label='frdir', cosz=cosz, albedo=albedo,
                            model=type, modeldir=modeldir, prefit_models_adir=prefit_models_adir)
        mydf_frdif = {x:y for x,y in zip(mydf.keys(), mydf.values()) if x in specific_predictors['frdif']}
        self.frdif = dem.model_predict_interp(mydf_frdif, label='frdif', cosz=cosz, albedo=albedo,
                            model=type, modeldir=modeldir, prefit_models_adir=prefit_models_adir)
        mydf_fcoup = {x:y for x,y in zip(mydf.keys(), mydf.values()) if x in specific_predictors['fcoup']}
        self.fcoup = dem.model_predict_interp(mydf_fcoup, label='fcoup', cosz=cosz, albedo=albedo,
                            model=type, modeldir=modeldir, prefit_models_adir=prefit_models_adir)

        ###### # ONLY FOR FDIR FOR NOW
        if normalize_by_grid_ave:
            # print("correcting direct flux prediction based on average ...")
            if normalize_fracs is None:
                # mydf_aveterrain = {'elen': np.mean(self.elen), 'sian': np.mean(self.sian),
                #                    'tcfn': np.mean(self.tcf), 'svfn': np.mean(self.svf)}

                mydf_aveterrain = {'sian': np.mean(self.sian),
                                   'svfn': np.mean(self.svf)}
                # print('correction for hires fluxes: size matrix = {}'.format(np.size(self.fdir)))
                xoldmean = np.mean(self.fdir)
            else:
                mydf_aveterrain = {
                                   # 'elen': np.sum(normalize_fracs*self.elen),
                                   'sian': np.sum(normalize_fracs*self.sian),
                                   # 'tcfn': np.sum(normalize_fracs*self.tcf),
                                   'svfn': np.sum(normalize_fracs*self.svf)}
                # print('correction for tiled fluxes: size frac = {}, {}'.format(np.size(normalize_fracs), np.size(self.svf)))
                xoldmean = np.sum(normalize_fracs*self.fdir)

            # print(mydf_aveterrain)
            # print(normalize_grid_ave_modeldir)

            aveterrain_fdir = dem.model_predict_interp(mydf_aveterrain, label='fdir', cosz=cosz, albedo=albedo,
                                             model=type, modeldir=normalize_grid_ave_modeldir,
                                             prefit_models_adir=prefit_models_adir)

            # xold = self.fdir
            # xnew = rescale_preserving_minval(xold, Ybar=aveterrain_fdir, lowerbound=-1.0, weights=None, xol)


            # transformation to prescribe new average and preserve minimum value
            xold = self.fdir
            xoldmin = np.min(self.fdir)

            xlowerbound = -1.0 # for direct flux (=complete shade)
            xnewmean = aveterrain_fdir
            xnewmin = max(xlowerbound, xnewmean - (xoldmean - xoldmin))

            # xnew = (self.fdir - xmean)/(xmean-xmin)*(aveterrain_fdir-xmin) + aveterrain_fdir
            # self.fdir = xnew

            xnew = xnewmean + (xnewmean-xnewmin) * (xold - xoldmean)/(xoldmean-xoldmin)

            self.fdir = xnew

            # print('average value: before = {:.3f}, after = {:.3f}'.format(np.mean(xold), np.mean(xnew)))

        ######

        self.fluxes = {'fdir':self.fdir, 'fdif':self.fdif,
                       'frdir': self.frdir, 'frdif': self.frdif,
                       'fcoup':self.fcoup
                       }

    # def plot_map_field(self, var = 'svf'):
    #     if var=='svf': map = self.svf
    #     elif var=='tcf': map = self.tcf
    #     elif var=='ss': map = self.ss
    #     elif var=='sc': map = self.sc
    #     elif var=='fdir': map = self.fdir
    #     elif var=='fdif': map = self.fdif
    #     elif var=='frdir': map = self.frdir
    #     elif var=='frdif': map = self.frdif
    #     else: raise Exception('must plot a valid field!')
    #     # xx = np.linspace(map.minx, map.maxx, map.nx)
    #     # yy = np.flipud(np.linspace(map.miny, map.maxy, map.ny))
    #     # XX, YY = np.meshgrid(xx, yy)
    #     plt.figure()
    #     vmax = np.quantile(np.ravel(map), 0.98)
    #     vmin = np.quantile(np.ravel(map), 0.02)
    #     # plt.pcolormesh(XX, YY, map.data, vmin=vmin, vmax=vmax)
    #     plt.imshow(map, vmin=vmin, vmax=vmax)
    #     plt.colorbar(extend='both')
    #     # plt.savefig(outputdir)
    #     plt.show()
    #     return


class grid_tile_database:

    # def __init__(self, h5file, landdir=None, tiles2map = False, pckdir = None):
    def __init__(self, h5file, landdir=None, tiles2map = False):
        # if landdir is Provided, saves also the tiled maps
        # and potentially high-res terrain variables
        dbs = h5py.File(h5file, 'r')
        # print(dbs.keys())
        # print(dbs['grid_data']['tile:1,is:1,js:1'].keys())
        # print(dbs['grid_data']['tile:1,is:1,js:1']['metadata'].keys())
        self.tid = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['tid'][:] # from 1 to ntiles
        self.frac = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['frac'][:]
        # print(frac)
        assert np.isclose( np.sum(self.frac), 1.0 )
        self.tile = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['tile'][:] # from 0 to ntiles-1
        self.ntiles = len(self.tile)
        self.ttype = dbs['grid_data']['tile:1,is:1,js:1']['metadata']['type'][:]
        self.ntiles_only_soil = len(self.ttype[self.ttype==3])

        if 1 in self.ttype:
            self.has_glacier = True
        else:
            self.has_glacier = False

        if 2 in self.ttype:
            self.has_lake = True
        else:
            self.has_lake = False

        if 3 in self.ttype:
            self.has_soil = True
        else:
            self.has_soil = False

        # dbs_tvr_vars = ['sinssina', 'sinscosa', 'svf', 'tcf']
        # dbs_tvr_vars = ['sinssina', 'sinscosa', 'svf', 'tcf']
        # dbs_tvr_vars = ['sinssina', 'sinscosa', 'svf', 'tcf', 'radstelev'] # radstelev --> elen
        dbs_tvr_vars = ['sinssina', 'sinscosa', 'svf', 'tcf', 'radavelev'] # radstelev --> elen

        dbs_trv = { elem : np.zeros(self.ntiles) for elem in dbs_tvr_vars}
        for elem in dbs_tvr_vars:

            if self.has_glacier:
                dbs_trv[elem][self.ttype==1] = dbs['grid_data'][
                        'tile:1,is:1,js:1']['glacier'][elem][:]
            if self.has_lake:
                dbs_trv[elem][self.ttype==2] = dbs['grid_data'][
                        'tile:1,is:1,js:1']['lake'][elem][:]
            if self.has_soil:
                dbs_trv[elem][self.ttype==3] = dbs['grid_data'][
                        'tile:1,is:1,js:1']['soil']['tile_hlsp_{}'.format(elem)][:]

        self.svf = dbs_trv['svf']
        self.tcf = dbs_trv['tcf']
        self.sc = dbs_trv['sinscosa']
        self.ss = dbs_trv['sinssina']
        # self.elen = dbs_trv['radstelev']
        self.elen = dbs_trv['radavelev']
        # self.dem = dbs_trv['dem']

        self.ave_svf = np.sum(self.frac*self.svf)
        self.ave_tcf = np.sum(self.frac*self.tcf)
        self.ave_ss = np.sum(self.frac*self.ss)
        self.ave_sc = np.sum(self.frac*self.sc)
        self.ave_elen = np.sum(self.frac*self.elen)
        # self.ave_dem = np.sum(self.frac*self.dem)

        self.tiled_terrain = {'sc':self.sc,
                              'ss': self.ss,
                              'tcf': self.tcf,
                              'svf': self.svf,
                              'elen': self.elen
                              # 'dem_tiles': self.dem
                             }

        if landdir is not None:
            # self.read_highres_maps(landdir, pckdir=pckdir)
            self.read_highres_maps(landdir)
        if tiles2map and landdir is not None:
            self.map_tiles_to_map()
        return

    # def read_highres_maps(self, landdir, pckdir=None):
    def read_highres_maps(self, landdir):
        self.svf_hr_map = gdal_tools.read_data(os.path.join(landdir, 'svf_ea.tif'))
        self.tcf_hr_map = gdal_tools.read_data(os.path.join(landdir, 'tcf_ea.tif'))
        self.ss_hr_map = gdal_tools.read_data(os.path.join(landdir, 'sinssina_ea.tif'))
        self.sc_hr_map = gdal_tools.read_data(os.path.join(landdir, 'sinscosa_ea.tif'))
        self.dem_hr_map = gdal_tools.read_data(os.path.join(landdir, 'dem_ea.tif'))
        # self.elen_hr_map = gdal_tools.read_data(os.path.join(landdir, 'radstelev_ea.tif'))
        self.elen_hr_map = gdal_tools.read_data(os.path.join(landdir, 'radavelev_ea.tif'))
        self.tiles_hr_map = gdal_tools.read_data(os.path.join(landdir, 'tiles.tif'))
        # add elevation stdv:
        # self.elen_hr_map = deepcopy(self.dem_hr_map)
        # self.elen_hr_map.data = (self.elen_hr_map.data -
        #             np.mean(self.elen_hr_map.data) ) / np.std(self.elen_hr_map.data)

        self.hr_terrain = {'sc':self.sc_hr_map,
                           'ss': self.ss_hr_map,
                           'tcf': self.tcf_hr_map,
                           'svf': self.svf_hr_map,
                           'elen': self.elen_hr_map,
                           'dem': self.dem_hr_map
                           }

        # if pckdir is not None:
        #     # read tiles and their areal-average properties
        #     with open(os.path.join(pckdir, 'hrus.pck'), "rb") as hru_file:
        #         self.hrus_ea_map = pickle.load(hru_file)
        #         # print("shape of hru map = {}".format(self.hrus_ea_map.data))
        #         # print("shape of svf map = {}".format(self.svf_hr_map.data))
        #     with open(os.path.join(pckdir, 'hru_properties.pck'), "rb") as hru_file2:
        #         self.hruprops = pickle.load(hru_file2)

        # compute grid averages of high-resolution fields
        self.svf_hr_ave = np.mean(self.svf_hr_map.data)
        self.tcf_hr_ave = np.mean(self.tcf_hr_map.data)
        self.sc_hr_ave = np.mean(self.sc_hr_map.data)
        self.ss_hr_ave = np.mean(self.ss_hr_map.data)
        self.elen_hr_ave = np.mean(self.elen_hr_map.data)
        return

    def map_tiles_to_map(self):
        # requires landdir to be give when class init
        # Do not use these to compute spatial statistics,
        # as here I am mapping on a lat-lon grid (tiles.tif) while hr terrain
        # variables are in equal area proj.
        self.tcf_mapped = map_tiled_prop(self.tile, self.tcf, self.tiles_hr_map.data)
        self.svf_mapped = map_tiled_prop(self.tile, self.svf, self.tiles_hr_map.data)
        self.ss_mapped =  map_tiled_prop(self.tile, self.ss, self.tiles_hr_map.data)
        self.sc_mapped =  map_tiled_prop(self.tile, self.sc, self.tiles_hr_map.data)
        self.elen_mapped =  map_tiled_prop(self.tile, self.elen, self.tiles_hr_map.data)

        self.lon = np.linspace(self.tiles_hr_map.minx,
                         self.tiles_hr_map.maxx, self.tiles_hr_map.nx)
        # self.lat = np.linspace(self.tiles_hr_map.miny,
        #                  self.tiles_hr_map.maxy, self.tiles_hr_map.ny)
        self.lat = np.linspace(self.tiles_hr_map.maxy,
                               self.tiles_hr_map.miny, self.tiles_hr_map.ny) # flipud to match raster images for plotting
        self.LON, self.LAT = np.meshgrid(self.lon, self.lat)
        return


    def remap_tiles_to_equal_area(self, outdir=None, vars=('tcf', 'svf', 'ss', 'sc', 'elen')):
        # must provide a directory (outdir) where raster files will be stored
        # requires map_tiles_to_map already run
        # lat = self.yy
        # lon = self.xx
        # vars2remap = ['tcf', 'svf', 'ss', 'sc']
        # vars2remap = {'tcf':None, 'svf':None, 'ss':None, 'sc':None, 'elen':None}
        # vars2remap ={var:None for var in vars}
        os.system("mkdir -p {}".format( os.path.join(outdir, 'tile_mapped_rasters')))
        for myvar in vars:
            infile_ll = os.path.join(outdir, 'tile_mapped_rasters',  '{}_map_ll.tif'.format(myvar))
            outfile_ea = os.path.join(outdir, 'tile_mapped_rasters', '{}_map_ea.tif'.format(myvar))
            logfile_ea = os.path.join(outdir, 'tile_mapped_rasters', '{}_map_ea.log'.format(myvar))
            my_array = getattr(self, "{}_mapped".format(myvar))

            # rster.write_raster_WGS84_ezdev(infile_ll, my_array, self.lat, self.lon, nodata=np.nan)
            rster.write_raster_WGS84_ezdev(infile_ll, my_array, self.lat, self.lon, nodata=-9999)
            rster.write_ea(infile_ll, outfile_ea, logfile_ea, eares=90.0, interp='average')
            mydata_ea = gdal_tools.read_data(outfile_ea)
            # mydata_ea

            setattr(self, "{}_mapped_ea", mydata_ea)
            mydata_ea.data[mydata_ea.data < -9000.0] = np.nan
            # print('shape of remapped tiled image::', mydata_ea.data.shape)
            # myminval = np.nanmin(mydata_ea.data)
            # print('myvar = {}, np.min remapped = {}'.format(myvar, myminval))
            # vars2remap[myvar] = mydata_ea

        # self.svf_mapped_ea = vars2remap['svf']
        # self.tcf_mapped_ea = vars2remap['tcf']
        # self.sc_mapped_ea = vars2remap['sc']
        # self.ss_mapped_ea = vars2remap['ss']
        # self.elen_mapped_ea = vars2remap['elen']

            setattr(self, "{}_mapped_ea".format(myvar), mydata_ea)



        # get lat lon coords based on lat lon map
        self.lonea = np.linspace(self.tiles_hr_map.minx,
                              self.tiles_hr_map.maxx, self.svf_mapped_ea.nx)
        # self.latea = np.linspace(self.tiles_hr_map.miny,
        #                       self.tiles_hr_map.maxy, self.svf_mapped_ea.ny)
        self.latea = np.linspace(self.tiles_hr_map.maxy,
                                 self.tiles_hr_map.miny, self.svf_mapped_ea.ny)
        self.LONea, self.LATea = np.meshgrid(self.lonea, self.latea)
        return

    def plot_map_field(self, var = 'svf', outputdir = None):
        if var=='svf': map = self.svf_hr_map
        elif var=='tcf': map = self.tcf_hr_map
        elif var=='ss': map = self.ss_hr_map
        elif var=='sc': map = self.sc_hr_map
        elif var=='elen': map = self.elen_hr_map
        elif var=='svf_tiles': map = self.svf_mapped
        elif var=='tcf_tiles': map = self.tcf_mapped
        elif var=='ss_tiles': map = self.ss_mapped
        elif var=='sc_tiles': map = self.sc_mapped
        elif var=='elen_tiles': map = self.elen_mapped
        elif var=='tiles': map = self.tiles_hr_map
        else: raise Exception('must plot a valid field!')
        if var in ['svf', 'tcf', 'ss', 'sc', 'ss', 'tiles', 'elen']:
            # xx = np.linspace(map.minx, map.maxx, map.nx)
            # yy = np.flipud(np.linspace(map.miny, map.maxy, map.ny))
            # use always lat lon boundaries for plotting maps (taken from tiles lat-lon map here)
            xx = np.linspace(self.tiles_hr_map.minx, self.tiles_hr_map.maxx, map.nx)
            # yy = np.flipud(np.linspace(self.tiles_hr_map.miny, self.tiles_hr_map.maxy, map.ny))
            yy = np.linspace(self.tiles_hr_map.maxy, self.tiles_hr_map.miny, map.ny)
            map_data = map.data
        elif var in ['svf_tiles', 'tcf_tiles', 'ss_tiles', 'sc_tiles', 'elen_tiles']:
            xx =  np.linspace(self.tiles_hr_map.minx,
                              self.tiles_hr_map.maxx, self.tiles_hr_map.nx)
            # yy =  np.linspace(self.tiles_hr_map.miny,
            #                   self.tiles_hr_map.maxy, self.tiles_hr_map.ny)
            yy =  np.linspace(self.tiles_hr_map.maxy,
                              self.tiles_hr_map.miny, self.tiles_hr_map.ny)
            map_data = map
        else: raise Exception('must plot a valid field!')

        XX, YY = np.meshgrid(xx, yy)
        plt.figure()
        if var == 'tiles':

            tilesdata = map_data
            maxvalp = np.max(tilesdata).astype(int)
            tilesdata[tilesdata < 0] = np.nan
            # tilesdata = map.data.astype(int)
            unique_values = np.unique(tilesdata)
            # print( np.unique(tilesdata) )

            ### define the colorbar for discrete data
            # cmap = plt.cm.jet  # define the colormap
            # extract all colors from the .jet map
            # cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be grey
            # cmaplist[0] = (.5, .5, .5, 1.0)

            # create the new map
            # cmap = mpl.colors.LinearSegmentedColormap.from_list(
            #     'Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            # bounds = np.linspace(0, maxvalp, maxvalp+1)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            ### end discrete colormap


            plt.pcolormesh(XX, YY, tilesdata, cmap="Pastel1")
            # plt.pcolormesh(XX, YY, tilesdata, cmap=cmap, norm=norm)
            cbar = plt.colorbar()

        elif var in ['sc', 'ss', 'svf', 'tcf', 'elen']:
            vmax = np.quantile(np.ravel(map_data), 0.98)
            vmin = np.quantile(np.ravel(map_data), 0.02)
            plt.pcolormesh(XX, YY, map_data, vmin=vmin, vmax=vmax, cmap = 'jet')
            cbar = plt.colorbar(extend='both')
            # cbar.set_label(var)
        else:
            plt.pcolormesh(XX, YY, map_data, cmap = 'jet')
            cbar = plt.colorbar(extend='both')
            # cbar.set_label(var)

        cbar.set_label(var)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        if var in ["sc", "ss", 'tcf', 'svf', 'elen']:
            plt.title("{} high-res. field".format(var))
        else:
            plt.title("{}, ntiles = {}".format(var, self.ntiles))
        if outputdir is not  None:
            plt.savefig(outputdir)
            plt.close()
        else:
            plt.show()
        return

    def plot_hist_field(self, var='svf'):
        if var == 'svf':
            map = self.svf_hr_map
        elif var == 'tcf':
            map = self.tcf_hr_map
        elif var == 'ss':
            map = self.ss_hr_map
        elif var == 'sc':
            map = self.sc_hr_map
        else:
            raise Exception('must plot a valid field!')

        vmax = np.quantile(np.ravel(map.data), 0.98)
        vmin = np.quantile(np.ravel(map.data), 0.02)
        plt.figure()
        # plt.hist(np.ravel(map.data), bins=300)
        plt.hist(np.ravel(map.data), bins=300)
        plt.xlim(vmin, vmax)
        # plt.savefig(outputdir)
        plt.show()
        return


def plot_tiled_stats(da, outfigdir=None, var='terrains', model='lee', myinputkind=None,
                     myinputkind2=None, mystats = ("mean",'stdv', 'skew', 'kurt')):
    # var -> can be 'terrain' or 'fluxes'
    # da -> data xarray with tile-statistics
    # outfigdir -> directory where output figures will be stored

    # matplotlib.use('Qt5Agg') # dyn show plots
    # nvars =  len(da.coords['vars'])
    labels =  da.coords['labels'].values
    terrains =  da.coords['terrains'].values
    domains =  da.coords['domains'].values
    npvalues = da.coords['npvalues'].values
    # nlabels = len(labels)
    # nterrains = len(terrains)
    stats0 = da.coords['stats'].values
    # statistics to plot: all for now:
    # mystats = ["mean",'stdv', 'skew', 'kurt']
    # mystats = ["mean",'stdv', 'skew', 'kurt']
    stats = [ stat for stat in list(stats0) if stat in mystats]
    # stats_symb_all = [r'mean $\mu_x$', r'st. dev. $ \sigma_x$', r'skewness $\gamma_x$', r'kurtosis $\xi_x$']
    stats_symb_all = {'mean':r'mean $\mu_{t}/\mu_{HR}$', 'stdv':r'st. dev. $ \sigma_t / \sigma_{HR}$', 'skew':r'skewness $\gamma_t / \gamma_{HR}$', 'kurt':r'kurtosis $\xi_t / \xi_{HR}$'}
    stats_symb = [sst for iss, sst in enumerate(stats_symb_all.values()) if list(stats_symb_all.keys())[iss] in stats]
    nstats = len(stats)

    mik_vals = [int(x) for x in myinputkind if x.isdigit()]

    if myinputkind2 is not None:
        mik_vals_2 = [int(x) for x in myinputkind2 if x.isdigit()]
        NTILES_2 = [npv * np.prod( list(mik_vals_2)) for npv in npvalues]

    NTILES = [npv * np.prod( list(mik_vals)) for npv in npvalues]
    # mystat = 'skew'
    # vars = da.coords['vars'].values
    # FIRST LET'S PLOT TERRAIN:
    # mymodel = 'lee'
    if var == 'terrains':
        # vars = terrains
        # elen is not used as a clustering terrain variable, so no convergence study
        vars = [vi for vi in terrains if vi != 'elen']
        datp = da['tiled_terrains'].sel(inputkinds=myinputkind) # select later
        if myinputkind2 is not None:
            datp2 = da['tiled_terrains'].sel(inputkinds=myinputkind2) # select later
        highres = da['hires_terrains']
    elif var == "fluxes":
        # vars = labels
        vars = [vi for vi in labels if vi != 'elen']
        datp = da['tiled_fluxes'].sel(inputkinds=myinputkind)  # select later
        if myinputkind2 is not None:
            datp2 = da['tiled_fluxes'].sel(inputkinds=myinputkind2)  # select later
        highres = da['hires_fluxes']
    nvars = len(vars)
    vars_symb_all = {'sc':r'$c_{\phi_s}$', 'ss':r'$s_{\phi_s}$', 'svf':r'$V_d$', 'tcf':r'$C_t$', 'elen':r'$h_n$',
                     'fdir':r'$f_{dir}$', 'frdir':r'$f_{rdir}$', 'fdif':r'$f_{dif}$', 'frdif':r'$f_{rdif}$', 'fcoup':r'$f_{coup}$'}
    vars_symb = [v for iv, v in enumerate(vars_symb_all.values()) if list(vars_symb_all.keys())[iv] in vars]
    # print(vars)
    # print(vars_symb)
    # data to plot:
    fig, axes = plt.subplots(nrows=nvars, ncols=nstats, figsize = (5*nstats, 5*nvars))
    # plt.grid(True)
    for i, myvar in enumerate( vars ):
        for j, mystat in enumerate(  stats):
            myvar_symb = vars_symb[i]
            MARKERS = ['-*', '-^', '-o'] # fir first input kind
            MARKERS2 = ['--*', '--^', '--o'] # for second input kind
            # COLORS = ['darkblue', 'darkred', 'darkgreen']
            COLORS = ['blue', 'red', 'green']
            # axes[i, j].plot(da.coords['npvalues'], np.ones(len(da.coords['npvalues'])), '--k')
            # axes[i, j].plot(da.coords['npvalues'], np.zeros(len(da.coords['npvalues'])), '--k')
            for id, mydom in enumerate( domains ):

                if var=='terrains':
                    y_highres = highres.sel(domains=mydom, terrains=myvar, stats=mystat).values
                    y_tiled = datp.sel(domains=mydom, terrains=myvar, stats=mystat).values
                    if myinputkind2 is not None:
                        y_tiled_2 = datp2.sel(domains=mydom, terrains=myvar, stats=mystat).values

                    stdv_highres = highres.sel(domains=mydom, terrains=myvar, stats='stdv').values
                    # print("stdv check", mydom, myvar, mystat, stdv_highres)

                elif var=='fluxes':
                    y_highres = highres.sel(domains=mydom, labels=myvar, stats=mystat, models=model).values
                    y_tiled = datp.sel(domains=mydom, labels=myvar, stats=mystat, models=model).values
                    if myinputkind2 is not None:
                        y_tiled_2 = datp2.sel(domains=mydom, labels=myvar, stats=mystat, models=model).values
                    stdv_highres = highres.sel(domains=mydom, labels=myvar, stats='stdv', models=model).values
                    # print("stdv check", mydom, myvar, mystat, stdv_highres)
                print('y_highres = {}'.format(y_highres))
                my_linewidth = 3
                # print( (y_tiled - y_highres)/stdv_highres )
                if mystat == 'mean':
                    # axes[i, j].plot(npvalues, (y_tiled - y_highres)/stdv_highres, MARKERS[id], color=COLORS[id], label=mydom)
                    axes[i, j].plot(NTILES, (y_tiled - y_highres), MARKERS[id], color=COLORS[id], label=mydom, linewidth=my_linewidth)

                    if myinputkind2 is not None:
                        axes[i, j].plot(NTILES, (y_tiled_2 - y_highres), MARKERS2[id], color=COLORS[id], label=mydom, linewidth=my_linewidth)
                    axes[i, j].plot(NTILES, y_tiled, '-k', linewidth=2)
                    # axes[i, j].plot(npvalues, y_highres*np.ones(np.shape(npvalues)), '--k', linewidth=2)
                    axes[i, j].plot(NTILES, np.zeros(len(da.coords['npvalues'])), '--k')
                # elif mystat == 'stdv':
                #     # axes[i, j].plot(npvalues, (y_tiled - y_highres)/stdv_highres, MARKERS[id], color=COLORS[id], label=mydom)
                #     axes[i, j].plot(NTILES, (y_tiled / y_highres), MARKERS[id], color=COLORS[id], label=mydom, linewidth=my_linewidth)
                #     axes[i, j].plot(NTILES, np.ones(len(da.coords['npvalues'])), '--k')
                elif mystat in ['stdv', 'skew', 'kurt']:
                    # axes[i, j].plot(npvalues, (y_tiled - y_highres), MARKERS[id], color=COLORS[id], label=mydom)
                    axes[i, j].plot(NTILES, y_tiled / y_highres, MARKERS[id], color=COLORS[id], label=mydom, linewidth=my_linewidth)
                    if myinputkind2 is not None:
                        axes[i, j].plot(NTILES, y_tiled_2 / y_highres, MARKERS2[id], color=COLORS[id], linewidth=my_linewidth)
                    axes[i, j].plot(NTILES, np.ones(len(da.coords['npvalues'])), '--k')

            # y_highres_Nepal = da.loc[:, 'Nepal', myvar, mystat, 'highres']
            # y_highres_Peru = da.loc[:, 'Peru', myvar, mystat, 'highres']
            # y_highres_EastAlps = da.loc[:, 'EastAlps', myvar, mystat, 'highres']
            # axes[i, j].plot(da.coords['nhills'], np.ones(len(da.coords['nhills'])), '--k')
            # axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Nepal', myvar, mystat, 'tile'].values/y_highres_Nepal,    '-o', color='blue',  label='Nepal')
            # axes[i, j].plot(da.coords['nhills'], da.loc[:, 'Peru', myvar, mystat, 'tile'].values/y_highres_Peru,     '-*', color='red',   label = 'Peru')
            # axes[i, j].plot(da.coords['nhills'], da.loc[:, 'EastAlps', myvar, mystat, 'tile'].values/y_highres_EastAlps, '-^', color='green', label = 'EastAlps')
            #
            # if mystat != 'mean' and mystat != 'skew':
            #     axes[i, j].set_ylim(bottom=0.0)
            # else:
            #     axes[i, j].set_ylim([ min( )*1.1, max()*1.1 ])
            # axes[i, j].set_title('{}, {}'.format(myvar, mystat))
            axes[i, j].set_xscale('log')
            axes[i,j].grid(True)
            if i == 0 and j ==  0:
                # axes[i, j].legend(loc='lower right', ncol=3) # legend only on first plot
                axes[i, j].legend(loc='lower right', ncol=5, bbox_to_anchor=(3.0, 1.45))  # legend only on first plot
            if i == 0:
                axes[i, j].set_title(stats_symb[j]) # titles only on top panels

            if i != len(vars)-1:
                axes[i, j].xaxis.set_ticklabels([]) # x-ticklabels only on bottom row
            else:
                axes[i, j].set_xlabel(r'$n_{t}$')
            if j == 0:
                # axes[i, j].set_ylabel(r'$   \frac{{   {}_k - {}_{{HR}} }}{{  \sigma_{{HR}}  }}$'.format(myvar, myvar, myvar), rotation=0, labelpad=46)
                axes[i,j].set_ylabel(myvar_symb, rotation=0, labelpad=46)
    if var=='fluxes':
        if myinputkind2 is not None:
            plt.savefig(os.path.join(outfigdir, 'conv_tiles_2schemes_{}_dashed={}_{}_{}.png'.format(myinputkind, myinputkind2, var, model)), dpi = 300)
        else:
            plt.savefig(os.path.join(outfigdir, 'conv_tiles_{}_{}_{}.png'.format(myinputkind, var, model)), dpi = 300)
    elif var=='terrains':
        if myinputkind2 is not None:
            plt.savefig(os.path.join(outfigdir, 'conv_tiles_2schemes_{}_dashed={}_{}.png'.format(myinputkind, myinputkind2, var)), dpi=300)
        else:
            plt.savefig(os.path.join(outfigdir, 'conv_tiles_{}_{}.png'.format(myinputkind, var)), dpi=300)
    plt.close()
    return




def plot_4_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=None, plot_differences=True):

    allvars = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup', 'sian', 'tcf', 'svf']
    d0 = RES_ALL[NPV_2_PLOT[0]]
    # plot_differences = True

    # do this when using block averaging on high res data
    # aveblock_data = np.flipud(getattr(d0.hires_fluxes, "{}".format('sian')).data)
    aveblock_data = getattr(d0.hires_fluxes, "{}".format('sian')).data
    # aveblock_lat = np.linspace(np.min(d0.LATea), np.max(d0.LATea), np.shape(aveblock_data)[0])
    aveblock_lat = np.linspace(np.max(d0.LATea), np.min(d0.LATea), np.shape(aveblock_data)[0])
    aveblock_lon = np.linspace(np.min(d0.LONea), np.max(d0.LONea), np.shape(aveblock_data)[1])
    AVELON, AVELAT = np.meshgrid(aveblock_lon, aveblock_lat)

    for iv, myvar in enumerate(allvars):
        # bnd = BND[iv]
        up_bnd = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, "{}".format(myvar)).data, 0.95)
        do_bnd = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, "{}".format(myvar)).data, 0.05)
        # print('bounds for {} :{:.3f} - {:.3f}'.format(myvar, do_bnd, up_bnd))
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        # cm0 = axes[0,0].pcolormesh(d0.LONea, d0.LATea, np.flipud(getattr(d0.hires_fluxes, "{}".format(myvar)).data), cmap='jet', vmin = do_bnd, vmax = up_bnd)
        HRMA = np.array( getattr(d0.hires_fluxes, "{}".format(myvar)).data)
        cm0 = axes[0, 0].pcolormesh(AVELON, AVELAT, HRMA,
                                    cmap='jet', vmin=do_bnd, vmax=up_bnd)

        # divnorm0 = mpl.colors.DivergingNorm(vmin = do_bnd, vcenter=0.0, vmax = up_bnd)
        # divnorm0 = mpl.colors.TwoSlopeNorm(vmin = do_bnd, vcenter=0.0, vmax = up_bnd)
        if not plot_differences:
            data1 = getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data
            data2 = getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data
            data3 = getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data

            cm1 = axes[0, 1].pcolormesh(d0.LONea, d0.LATea, data1, cmap='jet',vmin=do_bnd, vmax=up_bnd)
            cm2 = axes[1, 0].pcolormesh(d0.LONea, d0.LATea, data2, cmap='jet',vmin=do_bnd, vmax=up_bnd)
            cm3 = axes[1, 1].pcolormesh(d0.LONea, d0.LATea, data3, cmap='jet',vmin=do_bnd, vmax=up_bnd)

            # divnorm1 = mpl.colors.TwoSlopeNorm(vmin=do_bnd, vcenter=0.0, vmax=up_bnd)
            # divnorm2 = mpl.colors.TwoSlopeNorm(vmin=do_bnd, vcenter=0.0, vmax=up_bnd)
            # divnorm3 = mpl.colors.TwoSlopeNorm(vmin=do_bnd, vcenter=0.0, vmax=up_bnd)
            savename_4plots = 'mapped_tiles'
        else:
            data1 = HRMA - getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data
            data2 = HRMA - getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data
            data3 = HRMA - getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data
            cm1 = axes[0, 1].pcolormesh(d0.LONea, d0.LATea, data1, cmap='bwr', vmin=-2, vmax=2)
            cm2 = axes[1, 0].pcolormesh(d0.LONea, d0.LATea, data2, cmap='bwr', vmin=-2, vmax=2)
            cm3 = axes[1, 1].pcolormesh(d0.LONea, d0.LATea, data3, cmap='bwr', vmin=-2, vmax=2)
            savename_4plots = 'mapped_differences'

            # divnorm1 = mpl.colors.TwoSlopeNorm(vmin=data1.min(), vcenter=0.0, vmax=data1.max())
            # divnorm2 = mpl.colors.TwoSlopeNorm(vmin=data2.min(), vcenter=0.0, vmax=data2.max())
            # divnorm3 = mpl.colors.TwoSlopeNorm(vmin=data3.min(), vcenter=0.0, vmax=data3.max())

        axes[0, 0].set_title('High res. field')
        axes[0, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[0]].ntiles_only_soil))
        axes[1, 0].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[1]].ntiles_only_soil))
        axes[1, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[2]].ntiles_only_soil))

        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        axes[1, 0].set_xlabel('Longitude')

        axes[0, 1].set_ylabel('Latitude')
        axes[0, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_xlabel('Longitude')

        # divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=10)
        # divnorm = colors.TwoSlopeNorm(vcenter=0.0)

        # cbar0 = fig.colorbar(cm0, ax=axes[0, 0], norm=divnorm0)
        # cbar1 = fig.colorbar(cm1, ax=axes[0, 1], norm=divnorm1)
        # cbar2 = fig.colorbar(cm2, ax=axes[1, 0], norm=divnorm2)
        # cbar3 = fig.colorbar(cm3, ax=axes[1, 1], norm=divnorm3)

        if myvar in ['fdir'] and do_bnd < -0.99 and not plot_differences:
            cbar0 = fig.colorbar(cm0, ax=axes[0, 0], extend='max')
            cbar1 = fig.colorbar(cm1, ax=axes[0, 1], extend='max')
            cbar2 = fig.colorbar(cm2, ax=axes[1, 0], extend='max')
            cbar3 = fig.colorbar(cm3, ax=axes[1, 1], extend='max')
        else:
            cbar0 = fig.colorbar(cm0, ax=axes[0, 0], extend='both')
            cbar1 = fig.colorbar(cm1, ax=axes[0, 1], extend='both')
            cbar2 = fig.colorbar(cm2, ax=axes[1, 0], extend='both')
            cbar3 = fig.colorbar(cm3, ax=axes[1, 1], extend='both')
        countp = 0
        for j in [0, 1]:
            for i in [0, 1]:
                axes[j, i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                                transform=axes[j, i].transAxes,
                                size=20, weight='bold')
                countp += 1
        plt.tight_layout()
        plt.savefig(os.path.join(outfigdir, '{}_{}.png'.format(savename_4plots, myvar)))
        plt.close()
    return




def plot_2tiles_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=None):
    # gf.plot_6_tiled_maps(RES_ALL, NPV_2_PLOT, outfigdir=outfigdir_mapped, plot_differences=True)

    # allvars = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup', 'sian', 'tcf', 'svf']
    # plot tiled configuration together with one variable, for different ntiles
    allvars = ['tcf']
    d0 = RES_ALL[NPV_2_PLOT[0]]
    d1 = RES_ALL[NPV_2_PLOT[1]]
    # plot_differences = True

    # do this when using block averaging on high res data
    # aveblock_data = np.flipud(getattr(d0.hires_fluxes, "{}".format('sian')).data)
    aveblock_data = getattr(d0.hires_fluxes, "{}".format('sian')).data
    # aveblock_lat = np.linspace(np.min(d0.LATea), np.max(d0.LATea), np.shape(aveblock_data)[0])
    aveblock_lat = np.linspace(np.max(d0.LATea), np.min(d0.LATea), np.shape(aveblock_data)[0])
    aveblock_lon = np.linspace(np.min(d0.LONea), np.max(d0.LONea), np.shape(aveblock_data)[1])
    AVELON, AVELAT = np.meshgrid(aveblock_lon, aveblock_lat)



    # iv = 0
    # myvar = allvars[0]

    for iv, myvar in enumerate(allvars):
        # bnd = BND[iv]
        up_bnd0 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data, 0.95)
        do_bnd0 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data, 0.05)
        up_bnd1 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data, 0.95)
        do_bnd1 = np.quantile(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data, 0.05)

        # up_bnd0 = np.max(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data)
        # do_bnd0 = np.max(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data)
        # up_bnd1 = np.max(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data)
        # do_bnd1 = np.max(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data)
        up_bnd = max(up_bnd0, up_bnd1)
        do_bnd = max(do_bnd0, do_bnd1)
        # print('bounds for {} :{:.3f} - {:.3f}'.format(myvar, do_bnd, up_bnd))
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        # cm0 = axes[0,0].pcolormesh(d0.LONea, d0.LATea, np.flipud(getattr(d0.hires_fluxes, "{}".format(myvar)).data), cmap='jet', vmin = do_bnd, vmax = up_bnd)
        # TILES0 = np.array(getattr(d0, "{}".format('tiles_ea_map')).data)
        TILES0 = np.array(getattr(d0, "{}".format('tiles_hr_map')).data)
        TILES1 = np.array(getattr(d1, "{}".format('tiles_hr_map')).data)
        VAR0 = np.array(getattr(d0, "{}".format(myvar)).data)
        VAR1 = np.array(getattr(d1, "{}".format(myvar)).data)

        aveblock_lat_tiles = np.linspace(np.max(d0.LAT), np.min(d0.LAT), np.shape(TILES0)[0])
        aveblock_lon_tiles = np.linspace(np.min(d0.LON), np.max(d0.LONea), np.shape(TILES0)[1])
        AVTLON, AVTLAT = np.meshgrid(aveblock_lon_tiles, aveblock_lat_tiles)
        # AVTLON, AVTLAT = np.meshgrid(d0.LON, d0.LAT)

        TILES0_INT = TILES0.astype(int)
        TILES0_INT[TILES0_INT<-999] = 0 # for colorbar purposes only
        TILES0[TILES0<-999] = np.nan # values to plot
        TILES1_INT = TILES1.astype(int)
        TILES1_INT[TILES1_INT<-999] = 0 # for colorbar purposes only
        TILES1[TILES1<-999] = np.nan # values to plot
        cmap_tiles_0 = cm.get_cmap('Pastel1', len(np.unique(TILES0_INT)))
        cmap_tiles_1 = cm.get_cmap('Pastel1', len(np.unique(TILES1_INT)))

        # mat = plt.pcolormesh(d0.LON, d0.LAT, HRMA, cmap=cmap, vmin=np.nanmin(HRMA) - 0.5, vmax=np.nanmax(HRMA) + 0.5)

        cm0t = axes[0, 0].pcolormesh(AVTLON, AVTLAT, TILES0,
                                     cmap=cmap_tiles_0, vmin=np.nanmin(TILES0) - 0.5, vmax=np.nanmax(TILES0) + 0.5)

        cm1t = axes[0, 1].pcolormesh(AVTLON, AVTLAT, TILES1,
                                     cmap=cmap_tiles_1, vmin=np.nanmin(TILES1) - 0.5, vmax=np.nanmax(TILES1) + 0.5)

        data1 = getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data
        data2 = getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data
        # data3 = HRMA - getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data
        cm1 = axes[1, 0].pcolormesh(d0.LONea, d0.LATea, data1, cmap='bwr', vmin=0, vmax=up_bnd)
        cm2 = axes[1, 1].pcolormesh(d0.LONea, d0.LATea, data2, cmap='bwr', vmin=0, vmax=up_bnd)
        # cm3 = axes[1, 1].pcolormesh(d0.LONea, d0.LATea, data3, cmap='bwr', vmin=-2, vmax=2)
        savename_4plots = 'mapped_2tiles'

        fig.colorbar(cm0t, ax=axes[0,0])
        fig.colorbar(cm1t, ax=axes[0,1])
        fig.colorbar(cm1, ax=axes[1,0], extend='max')
        fig.colorbar(cm2, ax=axes[1,1], extend='max')


        axes[0, 0].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[0]].ntiles_only_soil))
        axes[0, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[1]].ntiles_only_soil))
        axes[1, 0].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[0]].ntiles_only_soil))
        axes[1, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[1]].ntiles_only_soil))
        # axes[1, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[2]].ntiles_only_soil))

        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        axes[1, 0].set_xlabel('Longitude')

        axes[0, 1].set_ylabel('Latitude')
        axes[0, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_xlabel('Longitude')


        countp = 0
        for j in [0, 1]:
            for i in [0, 1]:
                axes[j, i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                                transform=axes[j, i].transAxes,
                                size=20, weight='bold')
                countp += 1
        plt.tight_layout()
        plt.savefig(os.path.join(outfigdir, '{}_{}.png'.format(savename_4plots, myvar)))
        # plt.savefig('1.png')
        plt.close()
    return


def plot_4_tiled_hist(RES_ALL, NPV_2_PLOT, outfigdir=None):

    allvars = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup', 'sian', 'tcf', 'svf']
    d0 = RES_ALL[NPV_2_PLOT[0]]

    # do this when using block averaging on high res data
    aveblock_data = getattr(d0.hires_fluxes, "{}".format('sian')).data
    aveblock_lat = np.linspace(np.max(d0.LATea), np.min(d0.LATea), np.shape(aveblock_data)[0])
    aveblock_lon = np.linspace(np.min(d0.LONea), np.max(d0.LONea), np.shape(aveblock_data)[1])
    AVELON, AVELAT = np.meshgrid(aveblock_lon, aveblock_lat)

    # no plot histograms of the differences
    for iv, myvar in enumerate(allvars):
        # bnd = BND[iv]
        up_bnd = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, "{}".format(myvar)).data, 0.95)
        do_bnd = np.quantile(getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, "{}".format(myvar)).data, 0.05)
        # print('bounds for {} :{:.3f} - {:.3f}'.format(myvar, do_bnd, up_bnd))
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        # cm0 = axes[0,0].pcolormesh(d0.LONea, d0.LATea, np.flipud(getattr(d0.hires_fluxes, "{}".format(myvar)).data), cmap='jet', vmin = do_bnd, vmax = up_bnd)
        HRMA = np.array( getattr(d0.hires_fluxes, "{}".format(myvar)).data )

        cm0 = axes[0, 0].pcolormesh(AVELON, AVELAT, HRMA, cmap='jet', vmin=do_bnd, vmax=up_bnd)
        axes[0, 1].hist( np.ravel( getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data - HRMA), density=True, bins=100, alpha=0.8, color = 'blue', label = 'diff')
        axes[1, 0].hist( np.ravel( getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data - HRMA), density=True, bins=100, alpha=0.8, color = 'blue', label = 'diff')
        axes[1, 1].hist( np.ravel( getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data - HRMA), density=True, bins=100, alpha=0.8, color = 'blue', label = 'diff')


        axes[0, 1].hist( np.ravel( getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data), density=True, bins=100, alpha=0.6, color = 'red', label='tiles')
        axes[1, 0].hist( np.ravel( getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data), density=True, bins=100, alpha=0.6, color = 'red', label='tiles')
        axes[1, 1].hist( np.ravel( getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data), density=True, bins=100, alpha=0.6, color = 'red', label='tiles')

        axes[0, 1].hist( np.ravel( HRMA), density=True, bins=250, alpha=0.4, color = 'green', label='high res.')
        axes[1, 0].hist( np.ravel( HRMA), density=True, bins=250, alpha=0.4, color = 'green', label='high res.')
        axes[1, 1].hist( np.ravel( HRMA), density=True, bins=250, alpha=0.4, color = 'green', label='high res.')

        axes[0, 0].set_title('High res. field')
        axes[0, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[0]].ntiles_only_soil))
        axes[1, 0].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[1]].ntiles_only_soil))
        axes[1, 1].set_title('$n_t = {}$ tiles'.format(RES_ALL[NPV_2_PLOT[2]].ntiles_only_soil))


        hist_up_bnd_0 = np.nanquantile(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data - HRMA, 0.99)
        hist_do_bnd_0 = np.nanquantile(getattr(RES_ALL[NPV_2_PLOT[0]].mappedt_fluxes, "{}".format(myvar)).data - HRMA, 0.01)
        hist_up_bnd_1 = np.nanquantile(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data - HRMA, 0.99)
        hist_do_bnd_1 = np.nanquantile(getattr(RES_ALL[NPV_2_PLOT[1]].mappedt_fluxes, "{}".format(myvar)).data - HRMA, 0.01)
        hist_up_bnd_2 = np.nanquantile(getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data - HRMA, 0.99)
        hist_do_bnd_2 = np.nanquantile(getattr(RES_ALL[NPV_2_PLOT[2]].mappedt_fluxes, "{}".format(myvar)).data - HRMA, 0.01)

        axes[0, 1].set_xlim(hist_do_bnd_0, hist_up_bnd_0)
        axes[1, 0].set_xlim(hist_do_bnd_1, hist_up_bnd_1)
        axes[1, 1].set_xlim(hist_do_bnd_2, hist_up_bnd_2)


        if myvar in ['fdir'] and do_bnd < -0.99:
            cbar0 = fig.colorbar(cm0, ax=axes[0, 0], extend='max')
        else:
            cbar0 = fig.colorbar(cm0, ax=axes[0, 0], extend='both')

        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('frequency')
        axes[1, 0].set_xlabel('value')

        axes[0, 1].set_ylabel('frequency')
        axes[0, 1].set_xlabel('value')
        axes[1, 1].set_ylabel('frequency')
        axes[1, 1].set_xlabel('value')

        axes[0, 1].legend()
        axes[1, 0].legend()
        axes[1, 1].legend()

        countp = 0
        for j in [0, 1]:
            for i in [0, 1]:
                axes[j, i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                                transform=axes[j, i].transAxes,
                                size=20, weight='bold')
                countp += 1
        plt.tight_layout()
        plt.savefig(os.path.join(outfigdir, 'histograms4_{}.png'.format(myvar)))
        plt.close()
    return


def plot_tiled_scatter_plots(RES_ALL, NPV_2_PLOT, outfigdir = None, scattering_aveblock = 6, do_aveblock = False, do_hexbin = False):
    ### DO SCATTER  PLOTS
    gridsize1 = 14
    vars = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']

    vars_names = [r'$f_{dir}$', r'$f_{dif}$', r'$f_{rdir}$', r'$f_{rdif}$', r'$f_{coup}$']
    nvars = len(vars)
    NHILLS = NPV_2_PLOT
    nnhills = len(NHILLS)


    fig, axes = plt.subplots(nrows=nvars, ncols=nnhills, figsize=(18, 22))
    countp = 0
    for j, myvar in enumerate(vars):
        myvarname = vars_names[j]
        for i, mynhill in enumerate(NHILLS):

            ntiles_soil = RES_ALL[NPV_2_PLOT[i]].ntiles_only_soil
            XA0 = np.array( getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, myvar) )
            YA0 = np.array( getattr(RES_ALL[NPV_2_PLOT[i]].mappedt_fluxes, myvar) )
            if do_aveblock:
                XA = np.ravel( dem.crop_and_average(XA0, buffer=0, average=True, aveblock=scattering_aveblock))
                YA = np.ravel( dem.crop_and_average(YA0, buffer=0, average=True, aveblock=scattering_aveblock))
            else:
                XA = np.ravel( XA0 )
                YA = np.ravel( YA0 )
            if do_hexbin:
                axes[j,i].hexbin( XA, YA, gridsize = gridsize1, cmap = plt.cm.Greens, bins='log')
            else:
                axes[j,i].plot( XA, YA, 'o')
            # axes[j,i].scatter( np.ravel(   getattr( RES_ALL[NPV_2_PLOT[0]].hires_fluxes, myvar).data ) ,
            #                   np.ravel(   getattr( RES_ALL[NPV_2_PLOT[i]].mappedt_fluxes, myvar).data ) ,
            #                    marker='d', c='r', s=35)
            # axes[j,i].scatter( np.nanmean(res_all[i]['hrt_pred_{}'.format(FLUX_TERMS[j])]),
            #                    np.nanmean(res_all[i]['hru_pred_{}'.format(FLUX_TERMS[j])]),
            #                    marker='d', c='r', s=35)
            # minx = np.nanmin( np.ravel(all_res[i]['map_{}'.format(myvar)]))
            # maxx = np.nanmax( np.ravel(all_res[i]['map_{}'.format(myvar)]))
            # miny = np.nanmin( np.ravel(all_res[i]['mappedtile_{}'.format(myvar)]))
            # maxy = np.nanmax( np.ravel(all_res[i]['mappedtile_{}'.format(myvar)]))
            minx = np.nanmin(XA); maxx = np.nanmax(XA)
            miny = np.nanmin(YA); maxy = np.nanmax(YA)
            # minlx = min(minx, miny)
            # maxlx = min(maxx, maxy)
            # axes[2,0].plot([minlx, maxlx], [minlx, maxlx], 'k')
            # axes[j,i].plot([miny, maxy], [miny, maxy], 'k')
            axes[j,i].plot([minx, maxx], [minx, maxx], 'k')
            if i==0:
                pad = 7
                axes[j, i].annotate('{}'.format(myvarname),
                                    xy=(0, 0.5), xytext=(-axes[j, i].yaxis.labelpad - pad, 0),
                                    xycoords=axes[j,i].yaxis.label, textcoords='offset points',
                                    size='large', ha='right', va='center')
            if j != len(vars) - 1:
                pass
                # axes[j, i].xaxis.set_ticklabels(
                #     [])  # x-ticklabels only on bottom row
            else:
                axes[j, i].set_xlabel('high-res. field')

            if i == 0:
                axes[j, i].set_ylabel('tiles')
            if j == 0:
                axes[j, i].set_title(r'$n_t$ = {} tiles'.format(ntiles_soil))
            axes[j,i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                           transform=axes[j,i].transAxes,
                           size=20, weight='bold')
            countp += 1
    plt.tight_layout()
    if do_hexbin:
        plt.savefig(os.path.join(outfigdir, 'res_hexbin_hrus_new.png'), dpi = 300)
    else:
        plt.savefig(os.path.join(outfigdir, 'res_scatter_hrus_new.png'), dpi=300)
    plt.close()
    return

def plot_tiled_histogram_plots(RES_ALL, NPV_2_PLOT, outfigdir=None, scattering_aveblock=6, do_aveblock=False):

    ### DO SCATTER  PLOTS
    gridsize1 = 14
    vars = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']
    vars_names = [r'$f_{dir}$', r'$f_{dif}$', r'$f_{rdir}$', r'$f_{rdif}$', r'$f_{coup}$']
    nvars = len(vars)
    NHILLS = NPV_2_PLOT
    nnhills = len(NHILLS)


    fig, axes = plt.subplots(nrows=nvars, ncols=nnhills, figsize=(18, 22))
    countp = 0
    for j, myvar in enumerate(vars):
        myvarname = vars_names[j]
        for i, mynhill in enumerate(NHILLS):
            # XA = np.ravel(getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, myvar).data)
            # YA = np.ravel(getattr(RES_ALL[NPV_2_PLOT[i]].mappedtll_fluxes, myvar).data )

            ntiles_soil = RES_ALL[NPV_2_PLOT[i]].ntiles_only_soil


            XA0 = np.array( getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, myvar) )
            YA0 = np.array( getattr(RES_ALL[NPV_2_PLOT[i]].mappedt_fluxes, myvar) )
            if do_aveblock:
                XA = np.ravel( dem.crop_and_average(XA0, buffer=0, average=True, aveblock=scattering_aveblock))
                YA = np.ravel( dem.crop_and_average(YA0, buffer=0, average=True, aveblock=scattering_aveblock))
            else:
                XA = np.ravel( XA0 )
                YA = np.ravel( YA0 )

            # XA0 = getattr(RES_ALL[NPV_2_PLOT[0]].hires_fluxes, myvar)
            # YA0 = getattr(RES_ALL[NPV_2_PLOT[i]].mappedt_fluxes, myvar)
            # print(XA0.shape)
            # print(YA0.shape)
            # XA = np.ravel( dem.crop_and_average(XA0, buffer=0, average=True, aveblock=scattering_aveblock))
            # YA = np.ravel( dem.crop_and_average(YA0, buffer=0, average=True, aveblock=scattering_aveblock))
            axes[j,i].hist( XA, bins=50, density=True, alpha=0.9, label='High res. field')
            axes[j,i].hist( YA, bins=50, density=True, alpha=0.5, label='Tiles')
            up_bnd = np.quantile(XA, 0.99)
            do_bnd = np.quantile(XA, 0.01)
            axes[j,i].set_xlim(do_bnd, up_bnd)
            if i==0:
                pad = 7
                axes[j, i].annotate('{}'.format(myvarname),
                                    xy=(0, 0.5), xytext=(-axes[j, i].yaxis.labelpad - pad, 0),
                                    xycoords=axes[j,i].yaxis.label, textcoords='offset points',
                                    size='large', ha='right', va='center')
            axes[j, i].set_xlabel('Value')
            if i == 0:
                axes[j, i].set_ylabel('Frequency')
            if j == 0:
                axes[j, i].set_title(r'$n_t$ = {} tiles'.format(ntiles_soil))
            axes[j,i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                           transform=axes[j,i].transAxes,
                           size=20, weight='bold')
            countp += 1
            axes[0,0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outfigdir, 'res_pdfs_hrus_new.png'), dpi = 300)
    plt.close()
    return