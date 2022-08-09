

from numba import jit
from geopy import distance
import re
from netCDF4 import Dataset

import os
import numpy as np
import pickle
from geospatialtools import gdal_tools
import matplotlib.pyplot as plt
import photon_mc_land

################################################################################

###################  PREPARE DEM DATASET FOR ANALYSIS ##########################


# read land (SRTM or synthetic landscale)

# read ground albedo (uniform of MODIS)

################################################################################


def read_flip_preproecssed_srtm(filename=None, datadir=None):
    # load digital elevation map
    # obtained from GFDL preproecssing
    x = None
    y = None
    Z = None
    return x, y, Z


def read_flip_complete_srtm(filename=None, datadir=None):
    # read, flip and complete periodic boudaries of a srtm dataset

    lats, lons, data = read_srtm_dataset(path=datadir,
                                            filename=filename, plot = False)

    xb, yb, Zb = flip_srtm_dataset(lons, lats, data)

    # compute distances between grid point centers [in meters]
    dx = distance.distance( (yb[0], xb[1]), (yb[0], xb[0])).km*1000.0
    dy = distance.distance( (yb[1], xb[0]), (yb[0], xb[0])).km*1000.0

    # compute coordinates in meters
    Lxb = np.arange(np.size(xb))*dx
    Lyb = np.arange(np.size(yb))*dy

    # fix the edges to obtain a perfectly periodic domain for photomc analysis
    x, y, Z = complete_periodic_edges(
        Lxb, Lyb, Zb, offsetx=10, offsety=10, plot=False)

    # TODO: return lats and lons as well (done)
    # TODO: add periodic edges to those as well?
    return x, y, lons, lats, Z


def read_srtm_dataset(path=None, filename=None, plot = False):
    '''-------------------------------------------------------------------------
    # using a Digital Elevation Model (DEM) at 90m resolution from the
    # SRTM (Shuttle Radar Topography Mission)
    # filename -> PATH AND TILE NAME
    # return data (lons * lats)
    INPUT::
        path -> local path to the folder with the files
        filename -> name of the SRTM file
        plot (False) -> if True, plot the DEM
    RETURN::
        lats -> N to S
        lons -> W to E
        data -> (nlats * nlons) nlats N -> S down, nlons W -> E right
    -------------------------------------------------------------------------'''

    if path == None:
        path = os.path.join('..', '..', '..', 'Documents', 'dem_datasets')
        print('read_srtm_dataset: using default path!')
    if filename == None:
        filename = 'N46E012.hgt'
        print('read_srtm_dataset: reading default dataset!')

    pathname = os.path.join(path, filename)
    siz = os.path.getsize(pathname)
    dim = int(np.sqrt(siz/2))
    assert dim*dim*2 == siz, 'Invalid file size'
    # READ BIG ENDIAN SIGNED INTEGERS (>i2) 16-bit binary format (2 BYTES)
    # elevation in m wrt WGS84 EGM 96
    data = np.fromfile(pathname, np.dtype('>i2'), dim*dim).reshape((dim, dim))
    Ny, Nx = np.shape(data)
    # dimdata = Nx
    data[data < - 10000] = 0 # fill the voids

    data = data.astype(np.int16)
    # Nx, Ny = self.elevations.shape
    cor = re.split('(?:S|N)|(?:E|W)|\.', filename)
    N = float(cor[1])
    E = float(cor[2])

    snwe = re.sub("\d+", "", filename).split('.')[0]
    print(snwe)
    if snwe[0] == 'S':
        N = - N
    if snwe[1] == 'W':
        E = - E
    lons = np.linspace(E, E + 1, Nx)
    lats = np.linspace(N + 1, N, Ny)
    # y = np.linspace(N, N + 1, Ny)
    # Xgor, Ygor = np.meshgrid(xor, yor)

    if plot:
        LONS, LATS = np.meshgrid(lons, lats)
        # site_coord = np.array([[12.559, 46.192], [12.475, 46.6]])
        # site_name = ['Barcis', 'Padola']
        plt.figure()
        ax = plt.gca()
        # cs = plt.contourf(Xg, Yg, data , vmin=0, vmax=4000, cmap = 'gist_earth')
        # cs = plt.contourf(Xg, Yg, data , cmap = 'gist_earth')
        # cs = ax.pcolormesh(Xg, Yg, data , cmap = 'gist_earth')
        cs = ax.pcolormesh(LONS, LATS, data, cmap='gist_earth', shading='auto')
        # ax.plot(site_coord[:, 0], site_coord[:, 1], 'r+', markersize=3)
        # for i, txt in enumerate(site_name):
        #     ax.annotate(txt, (site_coord[i, 0] + .01, site_coord[i, 1] - 0.01))
        cbar = plt.colorbar(cs)
        cbar.set_label('Elev. [m]')
        plt.xlabel('Longitiude (E)')
        plt.ylabel('Latitude (N)')
        plt.show()

    return lats, lons, data


@jit(nopython=True)
def generate_synthetic_terrain(nx=100, ny=100, dtype = 'bowl',
                               x0 = 12.0, xL = 13.0,
                               y0 = 46.0, yL = 47.0,
                               Zmin = 3.0, Zmax = 3000.0):
    '''-------------------------------------------------------------------------
    Generate a synthetic DEM dataset for photomc
    # note: must still run the periodic adjustment afterwards
    INPUT:
        nx = 100 # x dimension (vertical dimension) - WEST -> EAST
        ny = 100 # x dimension (horizontal dimension) - SOUTH -> NORTH
        dtype: type of surface. Can be one of the following:
        - 'bowl' - > for a paraboloid surface
        - 'siny'  -> for a sinusoidal-in-y surface
        - 'sinx'  -> for a sinusoidal-in-x surface
        - 'flat'  -> for a flat surface
    RETURN:
          x -> coordinates in vertical dimension (longitude)
          y -> coordinates in horizontal dimension (latitude)
          X, Y -> corresponding grids of shape nx*ny [REMOVED]
          Z -> DEM array nx*ny
    -------------------------------------------------------------------------'''
    x = np.linspace(x0, xL, nx)
    y = np.linspace(y0, yL, ny)
    # Y, X = np.meshgrid(y, x) [REMOVED]
    if dtype == 'flat':
        Z = np.ones((nx, ny))*Zmin
    elif dtype == 'bowl':
        centerpx = x[nx//2]
        centerpy = y[ny//2]
        Z = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                Z[i,j] = (x[i] - centerpx)**2 + (y[j] - centerpy)**2
        Zmax_temp = np.max(Z)
        Z = Z*(Zmax - Zmin)/Zmax_temp + Zmin

    elif dtype == 'sinx':
        L = (xL-x0)/5.0
        Z = np.zeros((nx, ny))
        for i in range(nx):
            Z[i,:] = (1.0 + np.sin(2.0*np.pi*x[i]/L)) * (Zmax - Zmin)/2.0 + Zmin
    elif dtype == 'siny':
        L = (yL-y0)/5.0
        Z = np.zeros((nx, ny))
        for j in range(ny):
            Z[:,j] = (1.0 + np.sin(2.0*np.pi*y[j]/L)) * (Zmax - Zmin)/2.0 + Zmin
    else:
        raise Exception('generate_synthetic_terrain: must provide valid dtype!')
    return x, y, Z


@jit(nopython=True)
def flip_srtm_dataset(xor, yor, data):
    '''-------------------------------------------------------------------------
    From SRTM geo coords to the ones used in photomc
    -------------------------------------------------------------------------'''
    nx = len(xor)
    ny = len(yor)
    # CONVERT COORD TO THE ONES USED IN PHOTOMC
    y = np.zeros(ny)
    Z = np.zeros((nx, ny)).astype(np.int16)
    for i in range(nx):
        for j in range(ny):
            Z[i,j] = data[ny-j-1,i]
    # x = xor
    for j in range(ny):
        y[j] = yor[ny-j-1]
    return xor, y, Z



@jit(nopython=True)
def subset_domain(x, y, data, NOLAT=47.00, SOLAT=46.95, WELON=12.0, EALON=12.05):
    clon = np.logical_and(x>WELON, x<EALON)
    clat = np.logical_and(y>SOLAT, y<NOLAT)
    xb = x[clon]
    yb = y[clat]
    # indx = np.arange(Nx)
    # indy = np.arange(Ny)
    ix0 = np.where(clon)[0][0]
    iy0 = np.where(clat)[0][0]
    ixL = np.where(clon)[0][-1]
    iyL = np.where(clat)[0][-1]
    # Zb = data[ np.ix_(indx[clon], indy[clat])]
    Zb = data[ix0:ixL+1, iy0:iyL+1]
    return xb, yb, Zb


@jit(nopython=True)
def complete_periodic_edges(xb, yb, Zb, offsetx=5, offsety=5, plot=False):

    # ADD PERIoODIC BOUNDS TO THE DEM
    # offsets (x-y) must be smaller than the lateral dimensions of Zb!
    # constant dx, dy needed
    # nbx = np.size(xb)
    # nby = np.size(yb)

    nbx = np.shape(xb)[0]
    nby = np.shape(yb)[0]
    nbxtot = nbx + offsetx
    nbytot = nby + offsety
    ZNB = np.zeros((nbx + offsetx, nby + offsety))
    ZNB[:nbx, :nby] = Zb[:, :]

    # INTERPOLATE FIRST BELOW (WEST BOUND)
    ZNB[-1, :nby] = Zb[0, :]  # WEST BOUND = EAST BOUND
    for i in range(nbx, nbx + offsetx - 1):
        # print(i)
        alphai = (i - nbx) / offsetx
        # print(alphai)
        for j in range(nby):
            ZNB[i, j] = (1 - alphai) * ZNB[nbx - 1, j] + alphai * ZNB[nbx + offsetx - 1, j]

    ZNB[:, -1] = ZNB[:, 0]  # NORTH BOUND = SOUTH BOUND
    for j in range(nby, nby + offsety - 1):
        # print(j)
        alphaj = (j - nby) / offsety
        # print(alphaj)
        for i in range(nbxtot):
            ZNB[i, j] = (1 - alphaj) * ZNB[i, nby - 1] + alphaj * ZNB[i, nby + offsety - 1]

    # complete coordinate arrays
    xnb = xb[0] + np.arange(nbxtot) * (xb[1] - xb[0])
    ynb = yb[0] + np.arange(nbytot) * (yb[1] - yb[0])
    # if plot:
    #     plt.figure()
    #     plt.imshow(ZNB)
    #     plt.show()
    #
    #     plt.figure()
    #     plt.plot(xb)
    #     plt.plot(xnb, '--')
    #     # plt.plot(xb)
    #     # plt.plot(xb)
    #     plt.show()
    return xnb, ynb, ZNB




# def init_atmosphere(atmdir='sample_atm_profiles'):
#
#     tt = np.fromfile( os.path.join(atmdir, 'tt'), dtype=float, sep = '\t') # optical depths [nlayers]
#     wc = np.fromfile( os.path.join(atmdir, 'wc'), dtype=float, sep = '\t') # single scatt. albedos [nlayers]
#     zz = np.fromfile( os.path.join(atmdir, 'zz'), dtype=float, sep = '\t') # elevation [nlevels]
#     pr = np.fromfile( os.path.join(atmdir, 'pr'), dtype=float, sep = '\t') # pressure [nlevels]
#     # add: read data from csv instead
#     # return dict with uniform values,
#     # for levels-type variables, skip lower level value
#     levels = np.arange(np.size(tt), 0, -1)  # level numbering
#
#     nlayers = np.size(tt)
#     nlevels = np.size(tt) + 1
#     dz = np.zeros(nlayers)  # thickness of each layer
#     extb = np.zeros(nlayers)  # average extinction coeff contribution of each layer
#
#     # extb[0] = tt[0]
#     for i in range(1, nlayers):
#         dz[i - 1] = zz[i - 1] - zz[i]
#         extb[i] = (tt[i] - tt[i - 1]) / dz[i - 1]
#     # now fill in first and last value respectively
#     extb[0] = tt[0] / dz[0]
#     dz[-1] = zz[-2] - zz[-1]
#     # return zz, pr, tt, wc, extb, dz, levels
#
#
#
#     atm_dict = Dict.empty(
#         key_type=types.unicode_type,
#         value_type=types.float64[:],
#     )
#     atm_dict["zz"]     = np.asarray(zz       , dtype = np.float64)
#     atm_dict["pr"]     = np.asarray(pr       , dtype = np.float64)
#     atm_dict["tt"]     = np.asarray(tt       , dtype = np.float64)
#     atm_dict["wc"]     = np.asarray(wc       , dtype = np.float64)
#     atm_dict["extb"]   = np.asarray(extb     , dtype = np.float64)
#     atm_dict["dz"]     = np.asarray(dz       , dtype = np.float64)
#     atm_dict["levels"] = np.asarray(levels   , dtype = np.float64)
#     # atm_dict = {'zz':zz[:-1], 'pr':pr[:-1],
#     #             'tt':tt, 'wc':wc, 'extb':extb,
#     #             'dz':dz, 'levels':levels}
#     return atm_dict


# @jit(nopython=True)
# def get_atm_value(z, param='wc', dfatm=None):
#     # supported values => 'wc', 'tt', 'pr', 'zz'.
#     # get the value of an atmospheric parameter
#     # at the specified elevation level [m over msl]
#
#     if param in ['zz', 'pr']: # layers not levels,
#         # in this case skip first value i.e. return the lower value
#         vals = dfatm[param][1:]
#     else:
#         vals = dfatm[param]
#
#     if z >= np.max( dfatm['zz'] ):
#         value = vals[0] # second (lower) value for level values
#     elif z <= np.min( dfatm['zz'] ):
#         value = vals[-1]
#     else:
#         # indx = np.where( z < np.array(dfatm['zz']))[0][-1]
#         indx = np.where( z < dfatm['zz'])[0][-1]
#         value = vals[indx]
#     return value


def read_modis_albedos(type='summer', plot=False, datadir = ''):
    # read summer or winter albedos
    # and return them in photomc coordinates

    # datadir = os.path.join('..', '..', '..',
    #                        'Documents', 'dem_datasets')
    modisdir = os.path.join(datadir, 'modis_brdf_data')
    if type == 'summer':
        filename = 'MCD43C3.A2020242.006.2020276061520.hdf'
    elif type == 'winter':
        filename = 'MCD43C3.A2021092.006.2021101072821.hdf'
    else:
        raise Exception('specify a valid MODIS file!')

    fh = Dataset(os.path.join(modisdir, filename), mode='r')

    bsa_all = fh['Albedo_BSA_shortwave']
    wsa_all = fh['Albedo_WSA_shortwave']

    (nya, nxa) = bsa_all.shape
    y_all = np.linspace(90.0, -90.0, nya)
    # y_all = np.linspace(-90.0, 90.0, nya)
    x_all = np.linspace(-180.0, 180.0, nxa)

    maxlon = 13.00
    minlon = 12.00
    maxlat = 47.00
    minlat = 46.00
    indx = np.where(np.logical_and(x_all > minlon, x_all < maxlon))[0]
    indy = np.where(np.logical_and(y_all > minlat, y_all < maxlat))[0]

    yalb = y_all[indy]
    xalb = x_all[indx]

    bsa = bsa_all[indy[0]:indy[-1] + 1, indx[0]:indx[-1] + 1].data
    wsa = wsa_all[indy[0]:indy[-1] + 1, indx[0]:indx[-1] + 1].data


    # plt.figure()
    # plt.imshow(wsa_all)
    # plt.show()

    bsap = bsa.copy()
    bsap[np.logical_or(bsap < 0, bsap > 1)] = np.nan
    wsap = wsa.copy()
    wsap[np.logical_or(wsap < 0, wsap > 1)] = np.nan

    # SET MISSING VALUES TO REGIONAL AVERAGE
    bsa_ave = np.nanmean(bsap)
    wsa_ave = np.nanmean(wsap)

    bsap[np.isnan(bsap)] = bsa_ave
    wsap[np.isnan(wsap)] = wsa_ave

    # CONSTRUCT ARRAY FOR PHOTOMC::
    WSA_PMC = np.fliplr(np.rot90(wsap))
    BSA_PMC = np.fliplr(np.rot90(bsap))
    yalb = np.flipud(yalb)
    Yalb, Xalb = np.meshgrid(yalb, xalb)

    # plt.figure()
    # plt.imshow(WSA_PMC)
    # plt.show()

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        c0 = axes[0].pcolormesh(Yalb, Xalb, WSA_PMC)
        c1 = axes[1].pcolormesh(Yalb, Xalb, BSA_PMC)
        axes[0].set_xlabel('Latitude N')
        axes[0].set_ylabel('Longitude E')
        axes[1].set_xlabel('Latitude N')
        axes[1].set_ylabel('Longitude E')
        axes[0].set_title('White-sky albedo')
        axes[1].set_title('Black-sky albedo')
        fig.colorbar(c0, ax=axes[0])
        fig.colorbar(c1, ax=axes[1])
        plt.tight_layout()
        plt.show()
        #
        # plt.figure(figsize=(3,3))
        # plt.plot(WSA_PMC.flatten(), BSA_PMC.flatten(), 'or')
        # plt.plot([0, 1], [0, 1], 'k')
        # plt.xlabel('White-sky albedo')
        # plt.ylabel('Black-sky albedo')
        # plt.tight_layout()
        # plt.show()

    return yalb, xalb, bsap, wsap


@jit(nopython=True)
def complete_periodic_edges_ea(xb, yb, lonb, latb, Zb, offsetx=5, offsety=5):
    # ADD PERIOODIC BOUNDS TO THE DEM
    # UPDATED INCLUDING LAT and LON coords, as needed
    # for the preprocessing in the case of EA dem datasets (GFDL preprocessing)
    # offsets (x-y) must be smaller than the lateral dimensions of Zb!
    nbx = np.shape(xb)[0]
    nby = np.shape(yb)[0]
    nbxtot = nbx + offsetx
    nbytot = nby + offsety
    ZNB = np.zeros((nbx + offsetx, nby + offsety))
    ZNB[:nbx, :nby] = Zb[:, :]
    # INTERPOLATE FIRST BELOW (WEST BOUND)
    ZNB[-1, :nby] = Zb[0, :]  # WEST BOUND = EAST BOUND
    for i in range(nbx, nbx + offsetx - 1):
        # print(i)
        alphai = (i - nbx) / offsetx
        # print(alphai)
        for j in range(nby):
            ZNB[i, j] = (1 - alphai) * ZNB[nbx - 1, j] + alphai * ZNB[nbx + offsetx - 1, j]
    ZNB[:, -1] = ZNB[:, 0]  # NORTH BOUND = SOUTH BOUND
    for j in range(nby, nby + offsety - 1):
        # print(j)
        alphaj = (j - nby) / offsety
        # print(alphaj)
        for i in range(nbxtot):
            ZNB[i, j] = (1 - alphaj) * ZNB[i, nby - 1] + alphaj * ZNB[i, nby + offsety - 1]
    # complete coordinate arrays
    xnb = xb[0] + np.arange(nbxtot) * (xb[1] - xb[0])
    ynb = yb[0] + np.arange(nbytot) * (yb[1] - yb[0])
    lonnb = lonb[0] + np.arange(nbxtot) * (lonb[1] - lonb[0])
    latnb = latb[0] + np.arange(nbytot) * (latb[1] - latb[0])
    return xnb, ynb, lonnb, latnb, ZNB



# def preprocess_land_dem(metadata, periodic_buffer = 10, crop = 0.25, eares = 90.0):
def preprocess_land_dem(metadata):



    datadir = metadata['datadir']
    exp_name = metadata['name']

    # datadir_in = os.path.join(datadir, "GFDL_preproc_dems")
    dem_data_input_subfolder = metadata['dem_data_input_subfolder']
    datadir_in = os.path.join(datadir, dem_data_input_subfolder)
    datadir_out = os.path.join(datadir, 'output_{}'.format(exp_name), "preprocessed_dems")
    # datadir_out = os.path.join(datadir, 'preprocessed_dems')
    os.system("mkdir -p {}".format(datadir_out))
    domain  = metadata['domain']

    # metadata_keys =
    if 'eares' in metadata.keys():
        eares = metadata['eares']
    else:
        eares = 90.0
        # print('Warning:: Land preprocessing:: eares not provided using default = {}'.format(eares))

    if 'periodic_buffer' in metadata.keys():
        periodic_buffer = metadata['periodic_buffer']
    else:
        periodic_buffer = 10
        # print('Warning:: Land preprocessing:: periodic_buffer not provided using default = {}'.format(periodic_buffer))

    if 'crop' in metadata.keys():
        crop = metadata['crop']
    else:
        crop = 0.25
        # print('Warning:: Land preprocessing:: crop not provided using default = {}'.format(crop))

    # MAYBE PASS THESE PARAMETERS AS INPUTS
    # cbufferx = 0.25; cbuffery = 0.25 # crop original lat-lon file of these fractions on each side
    cbufferx = crop; cbuffery = crop # crop original lat-lon file of these fractions on each side
    # peoffsetx = 10; peoffsety = 10 # complete periodic edges
    peoffsetx = periodic_buffer; peoffsety = periodic_buffer # complete periodic edges
    # eares = 90.0 # target resolution [m] of equal area map

    # READ DEM IN LATLON
    demfile = os.path.join( datadir_in, '{}_dem_latlon.tif'.format(domain))
    data = gdal_tools.read_data(demfile)
    # print(data.nx)
    # print(data.ny)
    # print(data.projection)
    # print(data.proj4)
    # print(data.nodata)
    # print(data.minx, data.maxx)
    # print(data.miny, data.maxy)
    # print('res:', data.resx, data.resy)
    # plt.figure()
    # plt.imshow(data.data, vmin = 0)
    # # #plt.imshow(  np.rot90(np.fliplr(np.flipud(data.data))), vmin = 0)
    # # plt.imshow(  np.rot90(data.data, k=-1), vmin = 0)
    # plt.show()

    # CROP DATASET TO DESIRED LATLON BOUNDING BOX
    # minx_crop = 12.2; maxx_crop = 12.8; miny_crop = 46.2; maxy_crop = 46.8
    minx_crop = data.minx + cbufferx; maxx_crop = data.maxx - cbufferx
    miny_crop = data.miny + cbuffery; maxy_crop = data.maxy - cbuffery

    infile_ll_cropped = os.path.join(datadir_in, '{}_dem_latlon.tif'.format(domain))
    outfile_ll_cropped = os.path.join(datadir_out, '{}_dem_latlon_cropped.tif'.format(domain))
    logfile_ll_cropped = os.path.join(datadir_out, '{}_dem_latlon_cropped.log'.format(domain))

    os.system('rm -f {} {}'.format(logfile_ll_cropped, outfile_ll_cropped))
    os.system("gdalwarp -te %.16f %.16f %.16f %.16f %s %s >& %s" % (
        minx_crop, miny_crop, maxx_crop, maxy_crop,
        infile_ll_cropped, outfile_ll_cropped, logfile_ll_cropped))

    data_ll_cropped = gdal_tools.read_data(outfile_ll_cropped)
    # plt.figure()
    # plt.imshow(data_ll_cropped.data, vmin = 0)
    # plt.show()


    # REPROJECT CROPPED DEM TO EQUAL AREA PROJECTION [MOLLEWEIDE]
    infile_ea_cropped = os.path.join( datadir_out, '{}_dem_latlon_cropped.tif'.format(domain))
    outfile_ea_cropped = os.path.join(datadir_out, '{}_dem_ea_cropped.tif'.format(domain))
    logfile_ea_cropped = os.path.join(datadir_out, '{}_dem_ea_cropped.log'.format(domain))
    eaproj = '+proj=moll +lon_0=%.16f +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'

    # USE ALWAYS 90m reaolution first to reproject the data:
    # in case, corsen it later to eares (if different) after cleaning boundaries with -9999
    # eares_90 = 90.0
    minlon = data_ll_cropped.minx; maxlon = data_ll_cropped.maxx
    lproj = eaproj % float((maxlon+minlon)/2)
    os.system("rm -f {} {}".format(outfile_ea_cropped, logfile_ea_cropped))
    os.system('gdalwarp -r average -dstnodata -9999 -tr %.16f %.16f -t_srs "%s" %s %s >& %s'
              % (eares, eares, lproj, infile_ea_cropped, outfile_ea_cropped, logfile_ea_cropped))

    data_ea_cropped = gdal_tools.read_data(outfile_ea_cropped)


    # GET COORDINATE AND ARRAY FOR CROPPED EA DEM
    nx = data_ea_cropped.nx
    ny = data_ea_cropped.ny

    # APPROXIMATE - GET LATLON COORDINATES FROM FILE WITH SAME ~ CROP BOUNDS
    lons0 = np.linspace(data_ll_cropped.minx, data_ll_cropped.maxx, data_ea_cropped.nx)
    lats0 = np.linspace(data_ll_cropped.miny, data_ll_cropped.maxy, data_ea_cropped.ny)
    # x0 = np.arange(data_ea_cropped.nx)*eares
    # y0 = np.arange(data_ea_cropped.ny)*eares
    #
    # print(data_ea_cropped.data.shape)
    # print(lons0.shape, lats0.shape)
    # print(x0.shape, y0.shape)

    # ROTATE MATRIX ACCORDING TO RMC CONVENTION
    Z0 =  np.rot90( data_ea_cropped.data, k=-1)

    # print(np.min(data_ea_cropped.data))
    # plt.figure()
    # # plt.imshow(data_ea_cropped.data, vmin = 0)
    # plt.imshow(Z0)
    # plt.show()

    # print(Z0)
    # FURTHER CROP TO GET RID OF NODATA AT THE BOUNDARIES
    nbx, nby = np.shape(Z0)
    # print('shape of Z0 = {}'.format( np.shape(Z0)))
    # these buffers may need to be increased in different domain
    bfdx = 12 # EA min buffer to avoid missing data (EA:11, PE:6, NEP:7, FREA: 8)
    # bfdx = 0 # EA min buffer to avoid missing data (EA:11, PE:6, NEP:7, FREA: 8)
    bfdy = 0 # No need to crop in y for all
    Z = Z0[bfdx:nbx-bfdx, bfdy:nby-bfdy].copy()
    # x = x0[bfdx:nbx-bfdx].copy()
    # y = y0[bfdy:nby-bfdy].copy()
    lons = lons0[bfdx:nbx-bfdx].copy()
    lats = lats0[bfdy:nby-bfdy].copy()

    nbx2, nby2 = np.shape(Z)
    # print('shape of Z = {}'.format(np.shape(Z)))
    x = np.arange(nbx2).astype(float) * eares
    y = np.arange(nby2).astype(float) * eares

    # print('xxx', x[0], x[-1])
    # print('yyy', y[0], y[-1])

    # print(Z)

    # if np.min(Z) < -1000:
    #     raise Exception('Error: missing data in the scene. Buffer bfdx and bfdy may need to be increased!. Check the DEM.')


    # print('shape of Z = {}'.format( np.shape(Z)))



    # X, Y = np.meshgrid(x, y)
    # LONS, LATS = np.meshgrid(lons, lats)
    #
    # Y, X = np.meshgrid(y, x)
    # LATS, LONS = np.meshgrid(lats, lons)
    #
    # plt.figure()
    # plt.pcolormesh(LONS, LATS, Z)
    # # plt.pcolormesh(X, Y, Z)
    # plt.show()


    xnb, ynb, lonnb, latnb, ZNB = complete_periodic_edges_ea(
        x, y, lons, lats, Z, offsetx=peoffsety, offsety=peoffsety)


    # YNB, XNB = np.meshgrid(ynb, xnb)
    # LATNB, LONNB = np.meshgrid(latnb, lonnb)
    # plt.figure()
    # # plt.pcolormesh(LONNB, LATNB, ZNB, shading='nearest')
    # plt.pcolormesh(XNB, YNB, ZNB, shading='auto')
    # plt.show()

    # print( ZNB.shape, lonnb.shape, latnb.shape)
    # print( lonnb[0], lonnb[-1], latnb[0], latnb[-1])

    # PICKLE RESULTS TO READ THEM IN main_photon_cluster
    resdict = {'y':ynb, 'x':xnb, 'lats':latnb, 'lons':lonnb, 'Z':ZNB,
               'proj':"equal_area", 'eares':eares,
               'peoffsetx':peoffsetx, 'peoffesety':peoffsety
               }

    output_file = os.path.join(datadir_out, '{}_dem_input.pkl'.format(domain))
    with open(output_file, 'wb') as picklef:
        pickle.dump(resdict, picklef)

    # read back the saved file
    # with open(output_file, "rb") as input_file:
    #     resdict2 = pickle.load(input_file)

    # print(resdict['x'][0], resdict['x'][-1])
    # print(resdict['y'][0], resdict['y'][-1])
    # print(resdict['Z'][0, 0])
    # print(resdict2['x'][0], resdict2['x'][-1])
    # print(resdict2['y'][0], resdict2['y'][-1])
    # print(resdict2['Z'][0, 0])
    # exit()

    return