
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import numpy as np
from itertools import product
# from numba import jit
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pickle
import string
from math import trunc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error,  mean_absolute_percentage_error

from scipy import signal

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# import tensorflow as tf
#
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import preprocessing
from sklearn.ensemble import RandomForestRegressor


# print(tf.__version__)

from topocalc.viewf import viewf, gradient_d8
import matplotlib


def matplotlib_update_settings():
    # http://wiki.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    # this is a latex constant, don't change it.
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.8 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 12
    tick_size = inverse_latex_scale * 8

    # learn how to configure:
    # http://matplotlib.sourceforge.net/users/customizing.html
    params = {
        'axes.labelsize': text_size,
        'legend.fontsize': tick_size,
        'legend.handlelength': 2.5,
        'legend.borderaxespad': 0,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'font.size': text_size,
        'text.usetex': False,
        'figure.figsize': fig_size,
        # include here any neede package for latex
        # 'text.latex.preamble': [r'\usepackage{amsmath}',
        #                         ],
    }
    plt.rcParams.update(params)
    return


# @jit(nopython=True)
def stdv_elev(dem, x_pix_scale = 2, y_pix_scale=2,
              xdobounds=True, ydobounds=True):
    """-------------------------------------------------------------------------
     compute the elevation standard deviation for a given terrain map
    compute local standard deviation over blocks of size scale_npix
    in pixels. Repeat same values at the boundaries.
    -------------------------------------------------------------------------"""

    print('stdv_elev: This function should not be used')
    exit()

    xoffset1 = x_pix_scale // 2
    yoffset1 = y_pix_scale // 2
    xoffset2 = x_pix_scale - xoffset1
    yoffset2 = y_pix_scale - yoffset1
    # print('xoffset1={}'.format(xoffset1), 'xoffset2={}'.format(xoffset2))
    # print('yoffset1={}'.format(yoffset1), 'xoffset2={}'.format(yoffset2))
    (ny, nx) = np.shape(dem)

    if x_pix_scale >= nx and y_pix_scale >= ny:
        stelev = np.ones((ny, nx))*np.std(dem)
    else:
        stelev = np.zeros((ny, nx))
        for i in range(yoffset1, ny-yoffset2+1):
            for j in range(xoffset1, nx-xoffset2+1):
                # print(i,j)
                # print('y-bounds = ', i-yoffset1, i+yoffset2 )
                # print('x-bounds = ', j-xoffset1, j+xoffset2 )
                locmat = dem[i-yoffset1:i + yoffset2, j-xoffset1:j+xoffset2]
                # print(locmat)
                # print(np.shape(locmat))
                stelev[i,j] = np.std(locmat)

        ## FIX TOP AND BOTTOM BUFFERS::
        if ydobounds:
            for i in range(yoffset1):
                stelev[i,:] = stelev[yoffset1, :]
            for i in range(ny-yoffset2+1, ny):
                stelev[i, :] = stelev[ny - yoffset2, :]

        if xdobounds:
            for j in range(xoffset1):
                stelev[:, j] = stelev[:, xoffset1]
            for j in range(nx - xoffset2 + 1, nx):
                stelev[:, j] = stelev[:, nx - xoffset2]
    return stelev

# TESTIT:
# mysize = 12
# A = np.random.rand(mysize, mysize)
# # A = np.arange(mysize**2).reshape(mysize, mysize)
# B = stdv_elev(A, x_pix_scale = 5, y_pix_scale=6)
# print(B)
#
# plt.figure()
# plt.imshow(B)
# plt.show()




# @jit(nopython=True)
def jenness_area(dx, dy, Z):
    """-------------------------------------------------------------------------
    Compute the surface areas for a DEM following the method by Jenness, 2004
    At the boundaries copy the inner values.
    Z -> DEM
    dx, dy -> sized of the pixels along the 0th and 1st dimension of Z
    Note that units of dz, dy must be the same as those of the elevation Z
    -------------------------------------------------------------------------"""
    (nx, ny) = np.shape(Z)
    A = np.zeros((nx, ny))

    ntriangles = 8
    RXPOS = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    RYPOS = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    QXPOS = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
    QYPOS = np.array([1, 0, -1, -1, -1, 0, 1, 1])

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # extract local 3x3 matrix
            # Zloc = Z[i-1:i+2, j-1:j+2]
            # print(Zloc)
            P = Z[i, j]
            # position of the nearby central grid cell points relative to point P
            # ntriangles = 8
            # RXPOS = np.array([ 0, -1, -1,  -1,   0,   1,   1,   1])
            # RYPOS = np.array([ 1,  1,  0,  -1,  -1,  -1,   0,   1])
            # QXPOS = np.array([-1, -1, -1,   0,   1,   1,   1,   0])
            # QYPOS = np.array([ 1,  0, -1,  -1,  -1,   0,   1,   1])
            for k in range(ntriangles):
                R = Z[ (i + RXPOS[k]), (j + RYPOS[k]) ]
                Q = Z[ (i + QXPOS[k]), (j + QYPOS[k]) ]
                dx_PR = dx*(RXPOS[k])
                dy_PR = dy*(RYPOS[k])
                dx_PQ = dx*(QXPOS[k])
                dy_PQ = dy*(QYPOS[k])
                dx_RQ = dx*(QXPOS[k] - RXPOS[k])
                dy_RQ = dy*(QYPOS[k] - RYPOS[k])
                dPR = 0.5*np.sqrt(  dx_PR**2 + dy_PR**2 + (P-R)**2)
                dPQ = 0.5*np.sqrt(  dx_PQ**2 + dy_PQ**2 + (Q-P)**2)
                dQR = 0.5*np.sqrt(  dx_RQ**2 + dy_RQ**2 + (Q-R)**2)
                # print(dPR)
                # area of the triangle:
                s = 0.5*(dPR + dPQ + dQR)
                AT = np.sqrt(s*(s-dPR)*(s-dPQ)*(s-dQR))
                # print(AT)
                A[i, j] += AT

    # now fix the boudanries, duplicate inner values::
    A[:,0] = A[:,1]
    A[:,-1] = A[:,-2]
    A[0,:] = A[1,:]
    A[-1,:] = A[-2,:]
    return A

# # TEST WITH THE DATSET FROm JENNESS 2004
# dx = 100
# dy = 100
# Z = np.array([[190, 170, 155],
#               [183, 165, 145],
#               [175, 160, 122]])
#
# AR = jenness_area(dx, dy, Z)
# print(AR)

# first remove boundaries. e.g., the central half along each dimension

# @jit(nopython=True)
# ermove kwargs to use with Numba!
# def crop_and_average(Z, **kwargs):
#     # remove the outer part of an image, and coarsen then central part
#     # you can specify only one of offset or nsize!
#     # if 'offset' in kwargs:
#     #     offset = kwargs.get('offset')
#     #     print('offset = {}'.format(offset))
#     # if 'nsize' in kwargs:
#     #     nsize = kwargs.get('nsize')
#     #     print('nsize = {}'.format(nsize))
#     if 'aveblock' in kwargs:
#         aveblock = kwargs.get('aveblock')
#         # print('aveblock = {}'.format(aveblock))
#     else:
#         aveblock=1
#
#     if 'average' in kwargs:
#         average = kwargs.get('average')
#         # print('average = {}'.format(average))
#     else:
#         average=False
#
#     if 'buffer' in kwargs:
#         buffer = kwargs.get('buffer')
#         # print('buffer = {}'.format(buffer))
#     else:
#         buffer = 0 # by default, do not crop
#     #     average=False
#
#
#
#     # otherwise by default take offset equal to 1/4 the size of z
#     nx = Z.shape[0]
#     ny = Z.shape[1]
#
#     offset_x = np.int(np.floor(nx*buffer))
#     offset_y = np.int(np.floor(ny*buffer))
#     # offset_x = nx // 4
#     # offset_y = ny // 4
#
#     Zc = Z[offset_x:nx-offset_x, offset_y:ny-offset_y]
#     xn1 = Zc.shape[0]
#     yn1 = Zc.shape[1]
#
#     # get the largest matrix with sides multiple of ave_size
#     xn2 = (xn1 // aveblock ) * aveblock
#     yn2 = (yn1 // aveblock ) * aveblock
#
#     Zc2 = Zc[:xn2, :yn2]
#
#     if average: # coarsen the results:
#         # datatype = Zc2.dtype
#         # Zc3 = signal.convolve2d(Zc2, lave, boundary='symm', mode='same')
#         m, n = Zc2.shape
#         p, q = (aveblock, aveblock)  # Block size
#         b = Zc2.reshape(m // p, p, n // q, q).mean((1, 3), keepdims=1)
#         # Zc3 = np.repeat(np.repeat(b, (p), axis=(1)), (q), axis=3).reshape(Zc2.shape)
#         # Faster alternative::
#         Zc3 = np.empty((m // p, p, n // q, q), dtype=float)
#         # make it a float to compute averages!
#         Zc3[:] = b
#         Zc3.shape = Zc2.shape
#     else:
#         Zc3 = Zc2
#
#     if 'x' in kwargs and 'y' in kwargs:
#         x = kwargs.get('x')
#         y = kwargs.get('y')
#         # in this case crop the coordinate arrays too::
#         x2 = x[offset_x:nx-offset_x]
#         y2 = y[offset_y:ny - offset_y]
#         x3 = x2[:np.shape(Zc3)[0]]
#         y3 = y2[:np.shape(Zc3)[1]]
#         res = (x3, y3, Zc3)
#     else:
#         res = Zc3
#     return res



def crop_and_average(Z, **kwargs):
    # remove the outer part of an image, and coarsen then central part
    # you can specify only one of offset or nsize!
    # if 'offset' in kwargs:
    #     offset = kwargs.get('offset')
    #     print('offset = {}'.format(offset))
    # if 'nsize' in kwargs:
    #     nsize = kwargs.get('nsize')
    #     print('nsize = {}'.format(nsize))
    if 'aveblock' in kwargs:
        aveblock = kwargs.get('aveblock')
        # print('aveblock = {}'.format(aveblock))
    else:
        aveblock=1

    if 'average' in kwargs:
        average = kwargs.get('average')
        # print('average = {}'.format(average))
    else:
        average=False

    if 'buffer' in kwargs:
        buffer = kwargs.get('buffer')
        # print('buffer = {}'.format(buffer))
    else:
        buffer = 0 # by default, do not crop
    #     average=False

    # otherwise by default take offset equal to 1/4 the size of z
    nx = Z.shape[0]
    ny = Z.shape[1]

    offset_x = np.int(np.floor(nx*buffer))
    offset_y = np.int(np.floor(ny*buffer))
    # offset_x = nx // 4
    # offset_y = ny // 4

    Zc = Z[offset_x:nx-offset_x, offset_y:ny-offset_y]
    xn1 = Zc.shape[0]
    yn1 = Zc.shape[1]

    # get the largest matrix with sides multiple of ave_size
    xn2 = (xn1 // aveblock ) * aveblock
    yn2 = (yn1 // aveblock ) * aveblock

    Zc2 = Zc[:xn2, :yn2]

    nbx = xn1 // aveblock
    nby = yn1 // aveblock


    if 'x' in kwargs and 'y' in kwargs:
        x = kwargs.get('x')
        y = kwargs.get('y')
        # in this case crop the coordinate arrays too::
        x2 = x[offset_x:nx-offset_x]
        y2 = y[offset_y:ny - offset_y]
        x3 = x2[:np.shape(Zc2)[0]]
        y3 = y2[:np.shape(Zc2)[1]]
    else:
        x3 = np.arange(xn2)
        y3 = np.arange(yn2)


    # if average: # coarsen the results:
    if average and aveblock > 1: # coarsen the results:
    # if average: # coarsen the results:
        da = xr.DataArray(
            Zc2,
            coords=[("x", x3), ("y", y3)])

        db = da.coarsen(x=aveblock, y=aveblock, boundary='exact').mean()
        Zc3 = db.data
        x3 = db.coords['x'].values
        y3 = db.coords['y'].values
    else:
        Zc3 = Zc2

    if 'x' in kwargs and 'y' in kwargs:
        res = (x3, y3, Zc3)
    else:
        res = Zc3
    return res








################################################################################
# TESTING crop_boundaries

# Z2 = crop_boundaries(Z, aveblock=3)
# Z3 = crop_boundaries(Z, aveblock=3, average=True)
#
# print(Z)
# print(Z2)
# print(Z3)
# print('Z2 shape = {}'.format(Z2.shape))
# print('Z3 shape = {}'.format(Z3.shape))
#
#
# Z = np.arange(110).reshape((5, 22))
# Z = np.random.randn(43*34).reshape(43, 34)
# # Z = np.arange(43*34).reshape(43, 34)
#
# plt.figure()
# plt.imshow(Z2)
# plt.show()
#
# plt.figure()
# plt.imshow(Z3)
# plt.show()
################################################################################


# TODO: remove
def load_terrain_vars(cosz=None, phi=None, buffer = 0.25,
                      do_average=False, aveblock=1,
                      datadir = None):
    print("WARNING - load_terrain_vars - THIS FUNCTION IS DEPRECATED, "
          "use load_static_terrain_vars instead")
    """-------------------------------------------------------------------------
    Extract the digital elevation model used from the result of a
    Radiation Monte Carlo (RMC) simulation and use it to compute
    terrain variables to be used a predictors for the parameterization.
    -------------------------------------------------------------------------"""

    print('reading terrain variables '
          'for cosz = {}, phi = {}'.format(cosz, phi))

    # read DEM from simulation output
    # use cases instead of index (it is the same)
    simdir = os.path.join(datadir, 'output_sim', "output_sim_3D")
    dfc = pd.read_csv(os.path.join(datadir, 'list_sim_cases_3D.csv'))

    cases = dfc['cases'].values

    # dfc.set_index('cases', inplace=True)

    # if JOBID is None and cosz is not None and phi is not None and adir is not None:
    # idx0 = dfc.index[np.logical_and(
    #     np.abs(dfc['cosz'].values - cosz) < 1e-8,
    #     np.logical_and(np.abs(dfc['phi'].values - phi) < 1e-8,
    #                    np.abs(dfc['adir'].values - adir) < 1e-8))]
    # idx = idx0[0]
    # if len(idx0) > 1:
    #     raise Exception('load_3d_fluxes Error: Multiple '
    #                     'simulations satisfy requirements!')

    ds_filename = os.path.join(simdir, 'photonmc_output_{}.nc'.format(cases[0]))
    ds = xr.open_dataset(ds_filename)
    # READ AND SAVE TERRAIN VARIABLES:
    # y0 = ds['lat'].values
    # x0 = ds['lon'].values
    if 'lat_meters' not in ds.keys():
        y0 = ds['lat'].values
        x0 = ds['lon'].values
    else:
        y0 = ds['lat_meters'].values
        x0 = ds['lon_meters'].values
    Z0 = ds['elev'].values
    x, y, Z = crop_and_average(Z0, buffer=buffer,
                    aveblock=aveblock, average=do_average, x=x0, y=y0)

    invy = np.flipud(y)
    dem_dx = np.abs(x0[1] - x0[0])
    dem_dy = np.abs(y0[1] - y0[0])
    dem_spacing = np.sqrt(dem_dx * dem_dy)

    # TO EA HERE

    svf0, tcf0 = viewf(Z0.astype(np.float64),
                       spacing=dem_spacing, nangles=16)
    # instead of rot, switch dx and dy  - faster here
    slope0, aspect0 = gradient_d8( np.rot90(Z0),
                                   dem_dx, dem_dy, aspect_rad=True)

    # np.testing.assert_array_equal(svf0 > 0, True)
    # np.testing.assert_array_equal(tcf0 > 0, True)

    # area0 = jenness_area(dem_dx, dem_dy, Z0)

    # total_area0 = np.sum(area0)

    # compute standard deviation over blocks of 10km x 10km
    # sde0 = stdv_elev(Z0, y_pix_scale=100, x_pix_scale=100)


    ############ this is awfully slow for large convolution windows ############
                      # 12 pix = ~ 1 km size
                      # 24 pix = ~ 2 km size
    # aveblock_std = 24 # 100 pix = ~ 10 km size
    # # window = np.ones((aveblock_std, aveblock_std), dtype=np.float32)
    # window = np.ones((aveblock_std, aveblock_std),
    #                          dtype=np.float32)/float(aveblock_std**2)
    # mu = signal.convolve2d(Z0, window, boundary='wrap', mode='same')
    # sqdiff = np.sqrt((Z0 - mu) ** 2)
    # sde0 = signal.convolve2d(sqdiff, window, boundary='wrap', mode='same')
    ############################################################################
    sde0 = np.ones(np.shape(Z0))*np.std(Z0)


    # import matplotlib
    # matplotlib.use('Qt5Agg') # dyn show plots
    # plt.figure()
    # plt.imshow(sde)
    # plt.show()


    # print(db.data.shape)
    # print(dbm.shape)
    # print("new stdv")


    # db = da.coarsen(x=aveblock_std, y=aveblock_std, boundary='exact').mean()
    # Zc3 = db.data
    # x3 = db.coords['x'].values
    # y3 = db.coords['y'].values
    #
    # plt.figure()
    # plt.imshow(stelev0)
    # plt.show()

    # np.mean(area0)
    # np.std(area0)

    # aspect0 = np.fliplr(np.rot90(aspect0,k=-1))
    # slope0 = np.fliplr(np.rot90(slope0, k=-1))


    aspect0 = np.rot90(aspect0,k=-1)
    slope0 = np.rot90(slope0, k=-1)

    # compute svf and tcf normalized by cos(slope)
    svfnorm = svf0/np.cos(slope0)
    tcfnorm = tcf0/np.cos(slope0)

    # aspect0 = np.pi - aspect0
    # plt.figure()
    # plt.imshow(aspect0)
    # # plt.imshow(Z0)
    # plt.colorbar()
    # plt.show()
    #
    # tanslope = np.tan(slope0)
    # cosslope = np.cos(slope0)
    # Ssp = tanslope * np.sin(aspect)
    # Scp = tanslope * np.cos(aspect)
    # Ssp = np.sin(slope0) * np.sin(aspect0)
    # Scp = np.sin(slope0) * np.cos(aspect0)
    # SVFc0 = svf / cosslope
    # TCFc0 = tcf / cosslope
    # SVFc0 = svf
    # TCFc0 = tcf
    # NOTE: fix angles here!!!
    # SIAc0 = outer_cosz + np.sqrt(1-outer_cosz**2) * (
    #         np.cos(outer_phi)*Scp + np.sin(outer_phi)*Ssp)





    # phi2 = ds.attrs['phi']
    # assert np.isclose(phi, phi2)

    sinz =  np.sqrt(1 - cosz ** 2)
    # SIAc0 = cosz + sinz* (np.cos(phi) * Scp + np.sin(phi) * Ssp)
    # slopeangle = np.arctan(slope0)
    # sia0 = np.sin(slope0)*sinz*  np.abs(np.cos(phi - aspect0)) + cosz*np.cos(slope0)
    # FIX MINUS SIGN DUE TO MY STUPID AZIMUTH CONVENTION
    # if phi > 0:
    #     phis = phi - np.pi
    # else:
    #     phis = phi + np.pi

    phis = - phi
    # sia0 = np.sin(slope0)*sinz*np.cos(phis - aspect0) + cosz*np.cos(slope0)
    # divide also by cosz for consistency??
    sia0 = np.sin(slope0)*sinz*np.cos(phis - aspect0) + cosz*np.cos(slope0) # divide by cos(slope0)
    sianorm = np.tan(slope0)*sinz*np.cos(phis - aspect0) + cosz # divide by cos(slope0)
    # sianorm = np.tan(slope0)*sinz*np.cos(phis - aspect0)/cosz + 1 # divide by cos(slope0)

    # normlized by cosz too
    # sianormz = np.tan(slope0)*sinz/cosz*np.cos(phis - aspect0) + 1.0 # divide by cos(slope0)


    SIA0 = crop_and_average(sia0,
          average=do_average, aveblock=aveblock, buffer = buffer)
    # SIAnormz = crop_and_average(sianormz,
    #       average=do_average, aveblock=aveblock, buffer = buffer)

    SIAnorm = crop_and_average(sianorm,
            average=do_average, aveblock=aveblock, buffer = buffer)
    SVFnorm = crop_and_average(svfnorm,
            average=do_average, aveblock=aveblock, buffer = buffer)
    TCFnorm = crop_and_average(tcfnorm,
            average=do_average, aveblock=aveblock, buffer = buffer)
    TCF0 = crop_and_average(tcf0,
            average=do_average, aveblock=aveblock, buffer = buffer)
    SVF0 = crop_and_average(svf0,
                           average=do_average, aveblock=aveblock, buffer = buffer)
    aspect = crop_and_average(aspect0,
            average=do_average, aveblock=aveblock, buffer = buffer)
    slope = crop_and_average(slope0,
            average=do_average, aveblock=aveblock, buffer = buffer)
    sde = crop_and_average(sde0,
        average=do_average, aveblock=aveblock, buffer = buffer)

    # for now, using domain averager standard deviation of elevation
    # stelev = np.ones(np.shape(slope)) * np.std(Z)

    # area = crop_and_average(area0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)

    # FOR NOW, STUDY FLUXES PER UNIT OF HORIZONTAL AREA!
    # thus, the ration of pixel area to map are is
    # areap_over_areamap = 1/np.size(stelev) # this excludes outter buffer

    res = {}
    res['SIA0'] = SIA0 # SIA angle cosine (not normalized)
    res['SIAnorm'] = SIAnorm # SIA normlized by cos(slope)
    # res['SIAnormz'] = SIAnormz # SIA normlized by cos(slope)*cosz
    res['SVFnorm'] = SVFnorm
    res['TCFnorm'] = TCFnorm
    res['TCF0'] = TCF0 # not divided by cos(slope)
    res['SVF0'] = SVF0 # not divided by cos(slope)
    res['aspect'] = aspect
    res['slope'] = slope
    res['sde'] = sde
    res['Z'] = Z
    res['x'] = x
    res['y'] = y
    res['invy'] = invy
    res['cosz'] = cosz
    res['phi'] = phi
    return res



def load_static_terrain_vars(buffer = 0.25,
                      do_average=False, aveblock=1,
                      datadir = None):
    """-------------------------------------------------------------------------
    Extract the digital elevation model used from the result of a
    Radiation Monte Carlo (RMC) simulation and use it to compute
    terrain variables to be used a predictors for the parameterization.
    # read only static variables here,
    # compute solar incident angles separately for efficiency
    -------------------------------------------------------------------------"""

    simdir = os.path.join(datadir, 'output_sim', "output_sim_3D")
    dfc = pd.read_csv(os.path.join(datadir, 'list_sim_cases_3D.csv'))
    cases = dfc['cases'].values
    ds_filename = os.path.join(simdir, 'photonmc_output_{}.nc'.format(cases[0]))
    ds = xr.open_dataset(ds_filename)

    # READ AND SAVE TERRAIN VARIABLES:
    if 'lat_meters' not in ds.keys():
        y0 = ds['lat'].values
        x0 = ds['lon'].values
    else:
        y0 = ds['lat_meters'].values
        x0 = ds['lon_meters'].values
    Z0 = ds['elev'].values
    x, y, Z = crop_and_average(Z0, buffer=buffer,
                aveblock=aveblock, average=do_average, x=x0, y=y0)
    invy = np.flipud(y)
    dem_dx = np.abs(x0[1] - x0[0])
    dem_dy = np.abs(y0[1] - y0[0])
    dem_spacing = np.sqrt(dem_dx * dem_dy)

    # COMPUTE SKY VIEW AND TERRAIN CONFIGUTAION
    svf0, tcf0 = viewf(Z0.astype(np.float64),
                       spacing=dem_spacing, nangles=16)
    # instead of rot, switch dx and dy  - faster here
    slope0, aspect0 = gradient_d8( np.rot90(Z0),
                                   dem_dx, dem_dy, aspect_rad=True)


    # np.testing.assert_array_equal(svf0 > 0, True)
    # np.testing.assert_array_equal(tcf0 > 0, True)

    # area0 = jenness_area(dem_dx, dem_dy, Z0)

    # total_area0 = np.sum(area0)

    # compute standard deviation over blocks of 10km x 10km
    # sde0 = stdv_elev(Z0, y_pix_scale=100, x_pix_scale=100)


    ############ this is awfully slow for large convolution windows ############
    # 12 pix = ~ 1 km size
    # 24 pix = ~ 2 km size
    # aveblock_std = 24 # 100 pix = ~ 10 km size
    # # window = np.ones((aveblock_std, aveblock_std), dtype=np.float32)
    # window = np.ones((aveblock_std, aveblock_std),
    #                          dtype=np.float32)/float(aveblock_std**2)
    # mu = signal.convolve2d(Z0, window, boundary='wrap', mode='same')
    # sqdiff = np.sqrt((Z0 - mu) ** 2)
    # sde0 = signal.convolve2d(sqdiff, window, boundary='wrap', mode='same')
    ############################################################################

    sde0 = np.ones(np.shape(Z0))*np.std(Z0)
    elen0 = (Z0 - np.mean(Z0)) / np.std(Z0)

    # np.max(sde0)
    # np.min(sde0)


    # import matplotlib
    # matplotlib.use('Qt5Agg') # dyn show plots
    # plt.figure()
    # plt.hist([1,1,3,7])
    # plt.show()


    # print(db.data.shape)
    # print(dbm.shape)
    # print("new stdv")


    # db = da.coarsen(x=aveblock_std, y=aveblock_std, boundary='exact').mean()
    # Zc3 = db.data
    # x3 = db.coords['x'].values
    # y3 = db.coords['y'].values
    #
    # plt.figure()
    # plt.imshow(stelev0)
    # plt.show()

    # np.mean(area0)
    # np.std(area0)

    # aspect0 = np.fliplr(np.rot90(aspect0,k=-1))
    # slope0 = np.fliplr(np.rot90(slope0, k=-1))


    aspect0 = np.rot90(aspect0,k=-1)
    slope0 = np.rot90(slope0, k=-1)

    # compute svf and tcf normalized by cos(slope)
    svfnorm = svf0/np.cos(slope0)
    tcfnorm = tcf0/np.cos(slope0)

    # aspect0 = np.pi - aspect0
    # plt.figure()
    # plt.imshow(aspect0)
    # # plt.imshow(Z0)
    # plt.colorbar()
    # plt.show()
    #
    # tanslope = np.tan(slope0)
    # cosslope = np.cos(slope0)
    # Ssp = tanslope * np.sin(aspect)
    # Scp = tanslope * np.cos(aspect)
    # Ssp = np.sin(slope0) * np.sin(aspect0)
    # Scp = np.sin(slope0) * np.cos(aspect0)
    # SVFc0 = svf / cosslope
    # TCFc0 = tcf / cosslope
    # SVFc0 = svf
    # TCFc0 = tcf
    # NOTE: fix angles here!!!
    # SIAc0 = outer_cosz + np.sqrt(1-outer_cosz**2) * (
    #         np.cos(outer_phi)*Scp + np.sin(outer_phi)*Ssp)





    # phi2 = ds.attrs['phi']
    # assert np.isclose(phi, phi2)

    # sinz =  np.sqrt(1 - cosz ** 2)
    # SIAc0 = cosz + sinz* (np.cos(phi) * Scp + np.sin(phi) * Ssp)
    # slopeangle = np.arctan(slope0)
    # sia0 = np.sin(slope0)*sinz*  np.abs(np.cos(phi - aspect0)) + cosz*np.cos(slope0)
    # FIX MINUS SIGN DUE TO MY STUPID AZIMUTH CONVENTION
    # if phi > 0:
    #     phis = phi - np.pi
    # else:
    #     phis = phi + np.pi

    # phis = - phi
    # sia0 = np.sin(slope0)*sinz*np.cos(phis - aspect0) + cosz*np.cos(slope0)
    # divide also by cosz for consistency??
    # sia0 = np.sin(slope0)*sinz*np.cos(phis - aspect0) + cosz*np.cos(slope0) # divide by cos(slope0)
    # sianorm = np.tan(slope0)*sinz*np.cos(phis - aspect0) + cosz # divide by cos(slope0)
    # sianorm = np.tan(slope0)*sinz*np.cos(phis - aspect0)/cosz + 1 # divide by cos(slope0)

    # normlized by cosz too
    # sianormz = np.tan(slope0)*sinz/cosz*np.cos(phis - aspect0) + 1.0 # divide by cos(slope0)


    # SIA0 = crop_and_average(sia0,
    #                         average=do_average, aveblock=aveblock, buffer = buffer)
    # SIAnormz = crop_and_average(sianormz,
    #       average=do_average, aveblock=aveblock, buffer = buffer)

    # SIAnorm = crop_and_average(sianorm,
    #                            average=do_average, aveblock=aveblock, buffer = buffer)
    SVFnorm = crop_and_average(svfnorm,
                               average=do_average, aveblock=aveblock, buffer = buffer)
    TCFnorm = crop_and_average(tcfnorm,
                               average=do_average, aveblock=aveblock, buffer = buffer)
    TCF0 = crop_and_average(tcf0,
                            average=do_average, aveblock=aveblock, buffer = buffer)
    SVF0 = crop_and_average(svf0,
                            average=do_average, aveblock=aveblock, buffer = buffer)
    # aspect = crop_and_average(aspect0,
    #                           average=do_average, aveblock=aveblock, buffer = buffer)
    # slope = crop_and_average(slope0,
    #                          average=do_average, aveblock=aveblock, buffer = buffer)
    sde = crop_and_average(sde0,
                           average=do_average, aveblock=aveblock, buffer = buffer)

    elen = crop_and_average(elen0,
                           average=do_average, aveblock=aveblock, buffer = buffer)

    # for now, using domain averager standard deviation of elevation
    # stelev = np.ones(np.shape(slope)) * np.std(Z)

    # area = crop_and_average(area0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)

    # FOR NOW, STUDY FLUXES PER UNIT OF HORIZONTAL AREA!
    # thus, the ration of pixel area to map are is
    # areap_over_areamap = 1/np.size(stelev) # this excludes outter buffer

    res = {}
    # res['SIA0'] = SIA0 # SIA angle cosine (not normalized)
    # res['SIAnorm'] = SIAnorm # SIA normlized by cos(slope)
    # res['SIAnormz'] = SIAnormz # SIA normlized by cos(slope)*cosz
    res['SVFnorm'] = SVFnorm
    res['TCFnorm'] = TCFnorm
    res['TCF0'] = TCF0 # not divided by cos(slope)
    res['SVF0'] = SVF0 # not divided by cos(slope)
    res['aspect_uncropped'] = aspect0
    res['slope_uncropped'] = slope0
    res['sde'] = sde
    res['elen'] = elen
    res['Z'] = Z
    res['x'] = x
    res['y'] = y
    res['invy'] = invy
    # res['cosz'] = cosz
    # res['phi'] = phi
    return res



def comp_solar_incidence_field(terrain_dict, cosz=None, phi=None,
                     do_average=None, aveblock=None, buffer = None):
    # terrain_dict -> dictionary with terrain variable fields
    # as computed in load_terrain_vars()
    # note: those fields are already cropped and averaged!
    aspect0 = terrain_dict['aspect_uncropped']
    slope0 = terrain_dict['slope_uncropped']
    sinz = np.sqrt(1 - cosz ** 2)
    phis = - phi

    sia0 = np.sin(slope0)*sinz*np.cos(phis - aspect0) + cosz*np.cos(slope0) # divide by cos(slope0)
    # sianorm = np.tan(slope0)*sinz*np.cos(phis - aspect0) + cosz # divide by cos(slope0)
    sianorm = np.tan(slope0)*sinz/cosz*np.cos(phis - aspect0) + 1.0 # divide by cos(slope0) and cosz

    aspect = crop_and_average(aspect0,
                average=do_average, aveblock=aveblock, buffer = buffer)
    slope = crop_and_average(slope0,
                average=do_average, aveblock=aveblock, buffer = buffer)
    SIA0 = crop_and_average(sia0,
                average=do_average, aveblock=aveblock, buffer = buffer)
    SIAnorm = crop_and_average(sianorm,
                average=do_average, aveblock=aveblock, buffer = buffer)

    res = {}
    res['aspect'] = aspect # SIA angle cosine (not normalized)
    res['slope'] = slope # SIA angle cosine (not normalized)
    res['SIA0'] = SIA0 # SIA angle cosine (not normalized)
    res['SIAnorm'] = SIAnorm # SIA normlized by cos(slope) and COSZ
    res['cosz'] = cosz
    res['phi'] = phi
    return res


def load_3d_fluxes(cosz=None, phi=None, adir=None,
                   FTOAnorm=1361.0, buffer = 0.25,
                   do_average=False, aveblock=1,
                   datadir = None):

    # updated version:
    # - must provide cosz, phi, adir

    # print('reading radiative fluxes over 3D terrain simulation '
    #       'for cosz = {}, phi = {}'.format(cosz, phi))

    simdir = os.path.join(datadir, 'output_sim', "output_sim_3D")
    dfc = pd.read_csv(os.path.join(datadir, 'list_sim_cases_3D.csv'))
    dfc.set_index('cases', inplace=True)

    # if JOBID is None and cosz is not None and phi is not None and adir is not None:
    # uses cases instead
    idx0 = dfc.index[np.logical_and(
        np.abs(dfc['cosz'].values - cosz) < 1e-8,
        np.logical_and(np.abs(dfc['phi'].values - phi) < 1e-8,
                       np.abs(dfc['adir'].values - adir) < 1e-8))]
    idx = idx0[0]
    if len(idx0) > 1:
        raise Exception('load_3d_fluxes Error: Multiple '
                        'simulations satisfy requirements!')

    ds_filename = os.path.join(simdir, 'photonmc_output_{}.nc'.format(idx))
    ds = xr.open_dataset(ds_filename)
    # print(ds_filename)
    #
    # y0 = ds['lat'].values
    # x0 = ds['lon'].values
    # Z0 = ds['elev'].values
    #
    # print('shapes of lat, lon, Z:')
    # print(np.shape(y0), np.shape(x0), np.shape(Z0))
    # print('****')
    # npixels0 = np.size(Z0)
    # aveblock = 8
    # do_average = True

    ############################################################################
    # READ AND SAVE TERRAIN VARIABLES:
    # x, y, Z = crop_and_average(Z0, buffer=buffer,
    #             aveblock=aveblock, average=do_average, x=x0, y=y0)


    # print('shapes of lat, lon, Z after cropping')
    # print(np.shape(y), np.shape(x), np.shape(Z))

    # npixels = np.size(Z)
    # (nx, ny) = np.shape(Z)
    # invx = np.flipud(x)
    # invy = np.flipud(y)
    #
    # dem_dx = np.abs(x0[1] - x0[0])
    # dem_dy = np.abs(y0[1] - y0[0])
    # dem_spacing = np.sqrt(dem_dx * dem_dy)
    # # better use equal area prouj here?
    # svf0, tcf0 = viewf(Z0.astype(np.float64),
    #                    spacing=dem_spacing, nangles=16)
    # # slope0, aspect0 = gradient_d8( np.flipud(np.rot90(Z0)), dem_dx, dem_dy, aspect_rad=True)
    # slope0, aspect0 = gradient_d8( np.rot90(Z0),
    #                                dem_dx, dem_dy, aspect_rad=True)

    # area0 = jenness_area(dem_dx, dem_dy, Z0)

    # total_area0 = np.sum(area0)

    # stelev0 = stdv_elev(Z0, y_pix_scale=100, x_pix_scale=100)
    #
    # plt.figure()
    # plt.imshow(stelev0)
    # plt.show()

    # np.mean(area0)
    # np.std(area0)

    # aspect0 = np.fliplr(np.rot90(aspect0,k=-1))
    # slope0 = np.fliplr(np.rot90(slope0, k=-1))


    # aspect0 = np.rot90(aspect0,k=-1)
    # slope0 =np.rot90(slope0, k=-1)
    #
    # svf0 = svf0/np.cos(slope0)
    # tcf0 = tcf0/np.cos(slope0)

    # aspect0 = np.pi - aspect0
    # plt.figure()
    # plt.imshow(aspect0)
    # # plt.imshow(Z0)
    # plt.colorbar()
    # plt.show()
    #
    # tanslope = np.tan(slope0)
    # cosslope = np.cos(slope0)
    # Ssp = tanslope * np.sin(aspect)
    # Scp = tanslope * np.cos(aspect)
    # Ssp = np.sin(slope0) * np.sin(aspect0)
    # Scp = np.sin(slope0) * np.cos(aspect0)
    # SVFc0 = svf / cosslope
    # TCFc0 = tcf / cosslope
    # SVFc0 = svf
    # TCFc0 = tcf
    # NOTE: fix angles here!!!
    # SIAc0 = outer_cosz + np.sqrt(1-outer_cosz**2) * (
    #         np.cos(outer_phi)*Scp + np.sin(outer_phi)*Ssp)

    # nphots = ds.attrs['nphotons']
    ftoanorm_simul = ds.attrs['ftoanorm']
    # print("ftoanorm simulation = {} ".format(ftoanorm_simul))

    # phi2 = ds.attrs['phi']
    # assert np.isclose(phi, phi2)

    # sinz =  np.sqrt(1 - cosz ** 2)
    # SIAc0 = cosz + sinz* (np.cos(phi) * Scp + np.sin(phi) * Ssp)
    # slopeangle = np.arctan(slope0)
    # sia0 = np.sin(slope0)*sinz*  np.abs(np.cos(phi - aspect0)) + cosz*np.cos(slope0)
    # FIX MINUS SIGN DUE TO MY STUPID AZIMUTH CONVENTION
    # if phi > 0:
    #     phis = phi - np.pi
    # else:
    #     phis = phi + np.pi

    # phis = - phi
    # # sia0 = np.sin(slope0)*sinz*np.cos(phis - aspect0) + cosz*np.cos(slope0)
    # sia0 = np.tan(slope0)*sinz*np.cos(phis - aspect0) + cosz # divide by cos(slope0)
    #
    # SIAc = crop_and_average(sia0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)
    # SVFc = crop_and_average(svf0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)
    # TCFc = crop_and_average(tcf0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)
    # aspect = crop_and_average(aspect0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)
    # slope = crop_and_average(slope0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)
    # stelev = crop_and_average(stelev0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)

    # area = crop_and_average(area0,
    #     average=do_average, aveblock=aveblock, buffer = buffer)

    # FOR NOW, STUDY FLUXES PER UNIT OF HORIZONTAL AREA!
    # thus, the ration of pixel area to map are is
    # areap_over_areamap = 1/np.size(stelev) # this excludes outter buffer

    # stelev = np.ones(np.shape(slope)) * np.std(Z)



    # plt.figure()
    # plt.imshow(area)
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(area0)
    # plt.colorbar()
    # plt.show()
    ############################################################################

    # get array of local fluxes (pixel by pixel, already in W/m^-2)
    ecoupN = crop_and_average(ds['ecoup'].values,
                average=do_average, aveblock=aveblock, buffer = buffer)
    edirN = crop_and_average(ds['edir'].values,
                average=do_average, aveblock=aveblock, buffer = buffer)
    erdirN = crop_and_average(ds['erdir'].values,
                average=do_average, aveblock=aveblock, buffer = buffer)
    edifN = crop_and_average(ds['edif'].values,
                average=do_average, aveblock=aveblock, buffer = buffer)
    erdifN = crop_and_average(ds['erdif'].values,
                average=do_average, aveblock=aveblock, buffer = buffer)

    # plt.figure()
    # plt.imshow(ds['edir'])
    # plt.colorbar()
    # plt.clim(0, 0.1)
    # plt.show()

    plotlast = False
    if plotlast:
        EDIR2 = (ds['edir'].values).astype(float)
        # EDIR2 = (ds['edir'].values).astype(float) + (ds['edif'].values).astype(float)
        EDIR2[EDIR2 < 1E-6] = np.nan
        plt.figure()
        # plt.imshow(Z, alpha = 0.6, extent=(x[0], x[-1], y[-1], y[0]))
        # plt.imshow(Z, alpha = 0.6)
        plt.pcolormesh(y0, x0, Z0, alpha=0.4, shading='flat')
        # cbar = plt.colorbar()
        # cbar.set_label('Elev. [m msl]')
        # cbar.set_label('Elev. [m msl]')
        plt.ylabel('x EAST [DEM grid points]')
        plt.xlabel('y NORTH [DEM grid points]')
        # plt.imshow(EDIR2[:nby, :nbx], cmap='jet')
        plt.pcolormesh(y0, x0, EDIR2)
        plt.colorbar()
        plt.title('cosz = {}, $\phi$ = {}, $\\alpha$ = {}, '
                  'noscatter'.format(cosz, phi, adir))
        plt.gca().invert_yaxis()
        outfigdir = os.path.join(datadir, 'outputfig')
        plt.savefig(os.path.join(outfigdir, 'impacts_direct.png'), dpi=300)
        # plt.show()
        plt.close()

    # get pixel-by-pixel fluxes in [Wm^-2] wrt TOA total flux
    # need to use original number of pix (Before cropping!)
    orig_npix = np.size( ds['ecoup'].values ) # (= area domain / area pixel)
    ecoupN = ecoupN * orig_npix / ftoanorm_simul * FTOAnorm * cosz
    edirN = edirN   * orig_npix / ftoanorm_simul * FTOAnorm * cosz
    edifN = edifN   * orig_npix / ftoanorm_simul * FTOAnorm * cosz
    erdirN = erdirN * orig_npix / ftoanorm_simul * FTOAnorm * cosz
    erdifN = erdifN * orig_npix / ftoanorm_simul * FTOAnorm * cosz

    # get average fluxes over the entire domain
    ecoup = np.mean(ecoupN)
    edir =  np.mean(edirN)
    edif =  np.mean(edifN)
    erdir = np.mean(erdirN)
    erdif = np.mean(erdifN)
    # esurf = ecoup + edir + edif + erdir + erdif # fractions of total absorbed

    # plt.figure()
    # plt.imshow(edirN)
    # plt.colorbar()
    # plt.show()

    Fdir = edir
    Fdif = edif
    Frdir = erdir
    Frdif = erdif
    Fcoup = ecoup

    FMdir = edirN
    FMdif = edifN
    FMrdir = erdirN
    FMrdif = erdifN
    FMcoup = ecoupN

    ############################################################################
    # JUST HERE: USING THE VALUES FOR THE ENTIRE DOMAIN <--- INCLUDING BUFFERS
    # TO EVALUATE RATIOS
    etoa = ds.attrs['etoa'] / ftoanorm_simul * FTOAnorm * cosz
    eabs = ds.attrs['eabs'] / ftoanorm_simul * FTOAnorm * cosz
    esrf = ds.attrs['esrf'] / ftoanorm_simul * FTOAnorm * cosz
    etot = etoa + eabs + esrf

    # print('3D fraction of downward photons')
    # print('3D absorbed by atmosphere = {}'.format(eabs  / etot) )
    # print('3D re-emitted from TOA = {}'.format(etoa   / etot) )
    # print('3D absorbed by surface = {}'.format(esrf / etot) )
    ############################################################################

    res_3d = {}
    res_3d['Fdir'] = Fdir # areal average fluxes in Wm-2
    res_3d['Fdif'] = Fdif
    res_3d['Frdir'] = Frdir
    res_3d['Frdif'] = Frdif
    res_3d['Fcoup'] = Fcoup

    res_3d['FMdir'] =  FMdir # matrices of local fluxes (pixel by pixel, Wm-2)
    res_3d['FMdif'] =  FMdif
    res_3d['FMrdir'] = FMrdir
    res_3d['FMrdif'] = FMrdif
    res_3d['FMcoup'] = FMcoup

    # res_3d['SIAc'] = SIAc
    # res_3d['SVFc'] = SVFc
    # res_3d['TCFc'] = TCFc
    # res_3d['aspect'] = aspect
    # res_3d['slope'] = slope
    # res_3d['stelev'] = stelev

    # res_3d['Z'] = Z
    # res_3d['x'] = x
    # res_3d['y'] = y
    # res_3d['invy'] = invy

    res_3d['cosz'] = cosz
    res_3d['phi'] = phi
    res_3d['adir'] = adir

    return res_3d





def load_pp_fluxes(cosz=None, adir=None, FTOAnorm=1361.0, buffer = 0.25,
                   do_average=False, aveblock=1,
                   datadir = None):
    '''-------------------------------------------------------------------------
    read the fluxes obtained from the plane-parallel simulation
    which are more or less constant in the domain
    FTOA = downward flux at the top of atmosphere ( hyp.
    -------------------------------------------------------------------------'''
    # print('reading the corresponding plane-parallel simulation '
    #       'for cosz = {}, adir = {}'.format(cosz, adir))

    simdir = os.path.join(datadir, 'output_sim', "output_sim_PP")
    dfc_pp = pd.read_csv(os.path.join(datadir, 'list_sim_cases_PP.csv'))
    dfc_pp.set_index('cases', inplace=True)

    # use cases instead
    idx_pp0 = dfc_pp.index[np.logical_and(
        np.abs(dfc_pp['cosz'].values - cosz) < 1e-8,
        np.abs(dfc_pp['adir'].values - adir) < 1e-8)]

    # print("idx_pp:")
    # print(idx_pp)
    idx_pp = idx_pp0[0]
    if len(idx_pp0) > 1:
        raise Exception('load_pp_fluxes Error: Multiple '
                        'simulations satisfy requirements!')

    ds_pp = xr.open_dataset(os.path.join(simdir,
                             'photonmc_output_{}.nc'.format(
                             idx_pp)))

    ecoup_ppN = ds_pp['ecoup'].values
    edir_ppN = ds_pp['edir'].values
    edif_ppN = ds_pp['edif'].values
    erdir_ppN = ds_pp['erdir'].values
    erdif_ppN = ds_pp['erdif'].values

    ftoanorm_simul = ds_pp.attrs['ftoanorm']
    nphotons_simul = ds_pp.attrs['nphotons']
    # print('PP simulation: total number of photons = {}'.format(nphotons_simul))

    # flux components received by the surface [W/m^-2]
    dir = np.sum(edir_ppN)   / ftoanorm_simul * FTOAnorm * cosz
    dif = np.sum(edif_ppN)   / ftoanorm_simul * FTOAnorm * cosz
    rdir = np.sum(erdir_ppN) / ftoanorm_simul * FTOAnorm * cosz
    rdif = np.sum(erdif_ppN) / ftoanorm_simul * FTOAnorm * cosz
    coup = np.sum(ecoup_ppN) / ftoanorm_simul * FTOAnorm * cosz

    # FTOA = FTOAnorm*cosz # total flux
    # FTOA = FTOAnorm  # total flux
    # arrays in number of photon (counts)
    # don't really need to do this here
    # ecoup_ppN = crop_and_average(ds_pp['ecoup'].values,
    #         average=do_average, aveblock=aveblock, buffer = buffer)
    # edir_ppN = crop_and_average(ds_pp['edir'].values,
    #         average=do_average, aveblock=aveblock, buffer = buffer)
    # edif_ppN = crop_and_average(ds_pp['edif'].values,
    #         average=do_average, aveblock=aveblock, buffer = buffer)
    # erdir_ppN = crop_and_average(ds_pp['erdir'].values,
    #         average=do_average, aveblock=aveblock, buffer = buffer)
    # erdif_ppN = crop_and_average(ds_pp['erdif'].values,
    #         average=do_average, aveblock=aveblock, buffer = buffer)

    # nphotons0_pp = ds_pp.attrs['nphotons']
    etoa_pp = ds_pp.attrs['etoa']  / ftoanorm_simul * FTOAnorm * cosz
    eabs_pp = ds_pp.attrs['eabs']  / ftoanorm_simul * FTOAnorm * cosz
    esrf_pp = ds_pp.attrs['esrf']  / ftoanorm_simul * FTOAnorm * cosz
    etot_pp = etoa_pp + eabs_pp + esrf_pp

    # print('PP fraction of downward photons')
    # print('PP absorbed by atmosphere = {}'.format(eabs_pp / etot_pp))
    # print('PP re-emitted from TOA = {}'.format(etoa_pp  / etot_pp))
    # print('PP absorbed by surface = {}'.format(esrf_pp/ etot_pp))

    res_pp = {}
    res_pp['Fdir_pp'] =  dir # fluxes in Wm^-2
    res_pp['Fdif_pp'] =  dif
    res_pp['Frdir_pp'] = rdir
    res_pp['Frdif_pp'] = rdif
    res_pp['Fcoup_pp'] = coup

    return res_pp




# def train_neuralnet(dataset, label='C', train_frac = 0.8, plot=False,
#                     activation = 'relu', save=False,
#                     savedir=None, savename = None):
#
#     print('dataset total size = {}'.format(dataset.shape[0]))
#     print('dataset total number of predictors = {}'.format(dataset.shape[1]-1))
#     if train_frac < 0.999:
#         print("Splitting dataset in training and testing")
#         train_dataset = dataset.sample(frac=train_frac, random_state=0)
#         test_dataset = dataset.drop(train_dataset.index)
#
#     else:
#         print('Warning: no cross validation. Entire sample used for training!')
#         train_dataset = dataset
#         test_dataset = dataset
#
#     # plt.figure()
#     # sns.pairplot(train_dataset[['A', 'B', 'C']], diag_kind='kde')
#     # plt.show()
#
#     # split features from labels
#     train_features = train_dataset.copy()
#     test_features = test_dataset.copy()
#     train_labels = train_features.pop(label)
#     test_labels =   test_features.pop(label)
#
#
#
#
#
#
#     # train_dataset.describe().transpose()[['mean', 'std']]
#     normalizer = preprocessing.Normalization()
#     normalizer.adapt(np.array(train_features))
#     print(normalizer.mean.numpy())
#
#     first = np.array(train_features[:1])
#     with np.printoptions(precision=2, suppress=True):
#         print('First example:', first)
#         print()
#         print('Normalized:', normalizer(first).numpy())
#
#     #
#     def build_and_compile_model(norm):
#         model = keras.Sequential([
#             norm,
#             # layers.Dense(16, activation=activation), # relu -> __/
#             layers.Dense(16, activation=activation),
#
#             # layers.Dense(2, activation='relu'),  # by defualt linear
#             # layers.Dense(64, activation='relu'),
#             layers.Dense(1)
#         ])
#         model.compile(loss='mean_absolute_error',
#                       optimizer=tf.keras.optimizers.Adam(0.001))
#         return model
#
#     dnn_model = build_and_compile_model(normalizer)
#     dnn_model.summary()
#
#
#     history = dnn_model.fit(
#         train_features, train_labels,
#         validation_split=0.20,
#         verbose=0, epochs=100)
#
#     # first layer
#     print(dnn_model.layers[0].weights)
#
#     for elem in dnn_model.layers[1].weights:
#         print(elem)
#     # print(dnn_model.layers[0].bias.numpy())
#     # print(dnn_model.layers[0].bias_initializer)
#
#
#
#     def plot_loss(history):
#         plt.figure()
#         plt.plot(history.history['loss'], label='loss')
#         plt.plot(history.history['val_loss'], label='val_loss')
#         plt.ylim([0, 5])
#         plt.xlabel('Epoch')
#         plt.ylabel('Error')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#
#
#
#     def plot_pred(var = 'A'):
#         myvar = train_features[var]
#         x = tf.linspace(np.min(myvar) - 0.5 * np.std(myvar),
#                         np.max(myvar) + 0.5 * np.std(myvar), 30)
#         y = dnn_model.predict(x)
#         plt.figure()
#         plt.scatter(train_features[var], train_labels, label='Data')
#         plt.plot(x, y, color='k', label='Predictions')
#         plt.xlabel(var)
#         plt.ylabel(label)
#         plt.legend()
#         plt.show()
#
#
#
#
#     test_predictions = dnn_model.predict(test_features).flatten()
#
#     def plot_scatter():
#         scatter = plt.figure()
#         a = plt.axes(aspect='equal')
#         plt.scatter(test_labels, test_predictions, s=0.2)
#         # plt.scatter(test_labels, yhat, marker="+", color='red')
#         plt.xlabel('True Values')
#         plt.ylabel('Predictions')
#         maxv = max( np.max(test_labels), np.max(test_predictions)) + 0.1
#         minv = min( np.min(test_labels), np.min(test_predictions)) - 0.1
#         lims = [minv, maxv]
#         plt.xlim(lims)
#         plt.ylim(lims)
#         _ = plt.plot(lims, lims, 'k')
#         plt.show()
#         return scatter
#
#     if plot:
#         plot_loss(history)
#         # plot_pred(var = 'A')
#         # plot_pred(var = 'B')
#         scatterplot = plot_scatter()
#
#     if save:
#         if not os.path.exists(savedir):
#             os.makedirs(savedir)
#         dnn_model.save( os.path.join(savedir, '{}.hdf5'.format(savename)))
#         if plot:
#             scatterplot.savefig( os.path.join(savedir,
#                     '{}.png'.format(savename)))
#
#     return dnn_model


def train_multlr(dataset, label='C', train_frac=0.8, plot=False,
                 save=False,
                 savedir=None, savename=None, testdataname = None):
    # train_frac = 0.8
    # dataset = dfred.copy()
    train_dataset = dataset.sample(frac=train_frac, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop(label)
    test_labels = test_features.pop(label)

    mlr_model = LinearRegression()
    mlr_model.fit(train_features, train_labels)
    test_pred = mlr_model.predict(test_features)

    # get variable importance
    # NOTE - THIS ASSUMES THAT VARIABLES HAVE BEEN NORMALIZED BEFOREHAND
    importance = mlr_model.coef_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()

    def plot_scatter():
        scatterplot = plt.figure()
        a = plt.axes(aspect='equal')
        plt.scatter(test_labels, test_pred, s=0.2)
        # plt.scatter(test_labels, yhat, marker="+", color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        maxv = max(np.max(test_labels), np.max(test_pred)) + 0.1
        minv = min(np.min(test_labels), np.min(test_pred)) - 0.1
        lims = [minv, maxv]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims, 'k')
        plt.show()
        return scatterplot

    if plot:
        scatterplot = plot_scatter()

    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(os.path.join(savedir, '{}.pickle'.format(savename)), 'wb') as pklf:
            pickle.dump(mlr_model, pklf)
        if plot:
            scatterplot.savefig( os.path.join(savedir,
                    '{}.png'.format(savename)))

        dftest = pd.DataFrame({'MC':test_labels, 'PRED':test_pred})
        dftest.to_csv( os.path.join(savedir, testdataname))
    return mlr_model


def lee_model_predict(df, label='fdir', cosz=1.0, albedo=0.1):
    # must be applied at 10km or larger
    # normalize by cosz outside
    if 'sian' in df.keys() and cosz < 0.99:
        solar_inc = df['sian']
        # solar_inc = df['sian']/cosz
    # if 'sia0' in df.keys() and cosz < 0.99:
    #     solar_inc = df['sia0'] / cosz
    elif 'sian' in df.keys() and cosz >= 0.99:
        solar_inc = 0.0
    # elif label in ['frdirn', 'frdifn', 'frdir', 'frdif']:
    #     solar_inc = np.nan # not needed in these cases
    # else:
    #     raise Exception("lee model predict Error: for cosz < 1"
    #                     " 'sian' predictor must be provided")
    #
    if 'svfn' in df.keys():
        sky_view = df['svfn']
    if 'tcfn' in df.keys():
        terrain_config = df['tcfn']

    # if not norm_by_coss:
    #     sky_view = df['svf0']
    #     terrain_config = df['tcf0']
    #     if 'sian' in df.keys() and cosz < 0.99: # remove sian not sia0 in case
    #         solar_inc = df['sia0'] / cosz

    stdev_elev = 1600 # set fixed value for now -
    COSZ = np.array([0.1, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
    # izen = np.argmin(np.abs(cosz - COSZ)) + 1  # first line doubled, to match tables
    izen = np.argmin(np.abs(cosz - COSZ))  # first line doubled
    # interpolation is done in the wrapper function:
    # so check that cosz passed here has an admissible value
    assert np.abs(cosz - COSZ[izen]) < 1E-6

    #   SVF       SIA        INTERCEPT
    coeff_dir = np.array([
        [2.045E+1, 6.792E-1, -2.103E+1],  # copied first line
        [2.045E+1, 6.792E-1, -2.103E+1],
        [1.993E+0, 9.284E-1, -2.911E+0],
        [5.900E-2, 9.863E-1, -1.045E+0],
        [5.270E-3, 9.942E-1, -9.995E-1],
        [2.977E-3, 9.959E-1, -9.990E-1],
        [2.977E-3, 9.959E-1, -9.990E-1],
        [8.347E-3, 0.0, -8.393E-3]
    ])
    #   SVF         TCF       INTERCEPT
    coeff_rdir = np.array([
        [2.351E-1, 1.590E-1, -2.332E-1],  # copied first line
        [2.351E-1, 1.590E-1, -2.332E-1],
        [1.368E-1, 1.642E-1, -1.358E-1],
        [1.254E-1, 1.653E-1, -1.247E-1],
        [1.274E-1, 1.635E-1, -1.267E-1],
        [1.314E-1, 1.623E-1, -1.307E-1],
        [1.359E-1, 1.620E-1, -1.352E-1],
        [-4.463E-6, 1.556E-1, 1.287E-3],
    ])

    #     STDELEV    SVF       SIA       INETRCEPT
    coeff_dif = np.array([
        [3.146E-7, 4.385E+0, 6.723E-3, -4.382E+0],  # copied first line
        [3.146E-7, 4.385E+0, 6.723E-3, -4.382E+0],
        [6.001E-7, 4.068E+0, 2.456E-2, -4.085E+0],
        [7.436E-7, 3.911E+0, 5.606E-2, -3.960E+0],
        [7.806E-7, 3.763E+0, 1.049E-1, -3.863E+0],
        [7.581E-7, 3.559E+0, 1.734E-1, -3.727E+0],
        [7.015E-7, 3.298E+0, 2.543E-1, -3.547E+0],
        [6.359E-7, 2.984E+0, 0.0, -2.984E+0],
    ])

    #   SVF         TCF       INTERCEPT
    coeff_rdif = np.array([
        [1.493E-1, 1.621E-1, -1.483E-1],  # copied first line
        [1.493E-1, 1.621E-1, -1.483E-1],
        [1.462E-1, 1.654E-1, -1.454E-1],
        [1.454E-1, 1.673E-1, -1.446E-1],
        [1.465E-1, 1.683E-1, -1.457E-1],
        [1.443E-1, 1.682E-1, -1.435E-1],
        [1.446E-1, 1.686E-1, -1.439E-1],
        [-3.427E-6, 1.576E-1, 1.199E-3]
    ])



    if label == 'fdir':
        ypred = (coeff_dir[izen, 0] * sky_view +
                 coeff_dir[izen, 1] * solar_inc + coeff_dir[izen, 2])
        # print('ypred = {}'.format(ypred))
        ypred = np.maximum(-1.0, ypred)
    elif label == 'frdir' or label == 'frdirn':
        ypred = (coeff_rdir[izen, 0] * sky_view +
                 coeff_rdir[izen, 1] * terrain_config + coeff_rdir[izen, 2])
        # we are predicting ypred / albedo for now
        # ypred = ypred * (albedo / 0.1)
        # ypred = ypred * ( 1.0 / 0.1)
    elif label == 'fdif':
        ypred = (coeff_dif[izen, 0] * stdev_elev +
                 coeff_dif[izen, 1] * sky_view +
                 coeff_dif[izen, 2] * solar_inc + coeff_dif[izen, 3])
        ypred = np.maximum(-1.0, ypred)

    elif label == 'frdif' or label == 'frdifn':
        ypred = (coeff_rdif[izen, 0] * sky_view +
                 coeff_rdif[izen, 1] * terrain_config + coeff_rdif[izen, 2])
        # we are predicting ypred / albedo for now
        # ypred = ypred * (albedo / 0.1)
        # ypred = ypred * ( 1.0 / 0.1)

    # flux component not available for this model
    elif label == 'fcoup' or label == 'fcoupn':
        ypred = np.zeros(np.shape(sky_view))

    else:
        raise Exception('Error: must provide a valid label!')

    # parameterization reported here was obtained for a albedo = 0.1
    # need to apply rescaling for reflected flux components
    if label in ['frdir', 'frdif']:
        ypred = ypred * (albedo / 0.1)
    if label in ['frdirn', 'frdifn']:
        ypred = ypred * (1.0 / 0.1)

    return ypred


def model_predict_interp(df, label='dir', cosz=1.0, albedo=0.1,
                            model = 'MLR', modeldir = None, prefit_models_adir = 0.3):
                            # prefit_model_domain = 'EastAlps', prefit_model_buffer = '0.35',
                            # prefit_model_aveblock = 6):

    # predict flux correction, interpolating over cosz values
    # same arguments and output as functions 'my_model_predict' and 'lee_model_predict'
    # model can be set to 'mlr' (multiple linear regression)
    #                     'rfr' (random forest regression)
    #                     'nlr' (nonlinear regression)
    #                     'lee' (ucla model)
    COSZ = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
    # cosz = 0.0
    # interpolate=True
    # if interpolate:
    lower_cosz0 = np.where(COSZ < cosz)[0]
    upper_cosz0 = np.where(COSZ >= cosz)[0]
    if len(lower_cosz0) == 0:
        lower_cosz = COSZ[0]
    else:
        lower_cosz = COSZ[lower_cosz0[-1]]
    if len(upper_cosz0) == 0:
        upper_cosz = COSZ[-1]
    else:
        upper_cosz = COSZ[upper_cosz0[0]]
    # print('cosz: my {:.2f}, lower {}, upper {}'.format(cosz, lower_cosz, upper_cosz))
    if np.abs(upper_cosz-lower_cosz) < 1E-6:
        lower_weight = 1.0
        upper_weight = 0.0
    else:
        upper_weight = (cosz - lower_cosz)/(upper_cosz - lower_cosz)
        lower_weight = (upper_cosz - cosz)/(upper_cosz - lower_cosz)
        assert np.isclose(lower_weight+upper_weight, 1.0)
    # print('lower weight = {:.3f}, upper weight = {:.3f}'.format(lower_weight, upper_weight))


    if model == 'LEE':
        if upper_weight > 0.0:
            upper_ypred = lee_model_predict(df, label=label, cosz=upper_cosz, albedo=albedo)
        if lower_weight > 0.0:
            lower_ypred = lee_model_predict(df, label=label, cosz=lower_cosz, albedo=albedo)
    else:
        if upper_weight > 0.0:
            upper_ypred = my_model_predict(df, label=label, cosz=upper_cosz, albedo=albedo,
                                           model=model, modeldir=modeldir,
                                           prefit_models_adir=prefit_models_adir)
        if lower_weight > 0.0:
            lower_ypred = my_model_predict(df, label=label, cosz=lower_cosz, albedo=albedo,
                                           model=model, modeldir=modeldir,
                                           prefit_models_adir=prefit_models_adir)

    if upper_weight > 0.0 and lower_weight > 0.0:
        ypred = lower_ypred*lower_weight + upper_ypred*upper_weight
    elif np.isclose(lower_weight, 0.0) and np.isclose(upper_weight, 1.0):
        ypred = upper_ypred
    elif np.isclose(lower_weight, 1.0) and np.isclose(upper_weight, 0.0):
        ypred = lower_ypred
    else:
        raise Exception('something is wrong here - check interpolation model weights!')
    return ypred


def my_model_predict(df, label='fdir', cosz=1.0, albedo=0.1,
                     model='MLR', modeldir = None,
                     prefit_models_adir = 0.3):
                     # prefit_model_domain = 'EastAlps', prefit_model_buffer = '0.35',
                     # prefit_model_aveblock = 6):
    # as for lee function:
    # df should be a dataframe of better dictionary
    # to allow for 0D, 1D and 2D output

    # model -> nlr, mlr, rfr
    # modeldir -> directory where pickled models are stored
    # prefit_models_adir -> albedo of the data used to train the models
    # cosz -> must match one of the saved models

    # get data in dataframe form for model evaluation
    sample_pred = list(df.keys())[1]
    # print( list(df.keys()))
    if isinstance(df, pd.DataFrame):
        # reading_df = True
        # print("reading a data frame")
        matndim = 1
        input_is_scalar = False

    elif isinstance(df, dict) and np.isscalar(df[sample_pred]):
        # print('case of dict with scalar values - scalar prediction')
        matndim = 1
        input_is_scalar = True
        df = pd.DataFrame(df, index = np.array([1]))
        # print(df)
    elif isinstance(df, dict) and not np.isscalar(df[sample_pred]):
        input_is_scalar = False
        # print("reading a dictionary")
        # print(df)
        # check if the element of the dictionary are numpy arrays - if scalars, convert to arrays
        # are_arrays = isinstance(P, (list, tuple, np.ndarray))
        # if np.isscalar(df['sian']):
        #     print('scalar elements!')
        #     print(df['sian'])
        #     print(df)
        #     df['sian'] = np.atleast_1d(df['sian'])
        #     df['elen'] = np.atleast_1d(df['elen'])
        #     df['svfn'] = np.atleast_1d(df['svfn'])
        #     df['tcfn'] = np.atleast_1d(df['tcfn'])
            # df = {np.atleast_1d(df[key]) for key in df.keys()}
            # df = {np.array(df[key]) for key in df.keys()}
        input_is_array = isinstance(df[sample_pred], (np.ndarray))
        if not input_is_array:
            raise Exception("The elements of df [predictors] must be either scalars of np arrays")
        matndim = df[sample_pred].ndim
        matshape = df[sample_pred].shape
        matsize = df[sample_pred].size

        if not matndim in [1,2]:
            raise Exception('Input must either be a df or a dict of scalars, of 1D arrays or of 2D arrays!')

        # print("sizes dict:: ndim = {}, size = {}, shape = {}".format(matndim, matsize, matshape))

    else:
        print('type of df = ', type(df))
        raise Exception("Input df must be either a dict or a data frame")

    # print("*")




# if matndim > 1:
    #     # take care of the possible presence of NaNs or -9999
    #     # list_nan_points = np.where( np.logical_or( np.isnan(mydf['svfn']), mydf['svfn'] < -9000))
    #     list_nan_points = np.where(np.logical_or(np.logical_or(np.isnan(df['svfn']), df['svfn'] < -9000),
    #                                              np.logical_not(np.isfinite(df['svfn']))))
    #     for fluxc in df.keys():  # set it to 1 now, mask it out afterwards
    #         df[fluxc][list_nan_points] = 1.0
    # print('type df = {}'.format(type(df)))

    if isinstance(df, dict):
        if input_is_array and matndim > 1:
            #####
            # for 2d maps only: if there are any missing points, mask them out and do not predict

            # if 'sian' in df.keys():
            #     list_nan_points = np.where(np.logical_or(np.logical_or(np.isnan(df['sian']), df['sian'] < -9000),
            #                                              np.logical_not(np.isfinite(df['sian']))))

            if 'elen' in df.keys():
                list_nan_points = np.where(np.logical_or(np.logical_or(np.isnan(df['elen']), df['elen'] < -9000),
                                                         np.logical_not(np.isfinite(df['elen']))))
            else:
                list_nan_points = np.where(np.logical_or(np.logical_or(np.isnan(df[sample_pred]), df[sample_pred] < -9000),
                                                          np.logical_not(np.isfinite(df[sample_pred]))))



            # def condition(df, myvarbb='sian'):
            #     return np.logical_or(np.logical_or(np.isnan(df[myvarbb]), df[myvarbb] < -9000.0),
            #                          np.logical_not(np.isfinite(df[myvarbb])))
            #
            # bool_sian = condition(df, myvarbb='sian')
            # bool_tcfn = condition(df, myvarbb='tcfn')
            # bool_svfn = condition(df, myvarbb='svfn')
            # bool_or = bool_sian * bool_tcfn * bool_svfn # logical or between 3 bool arrays
            # list_nan_points = np.where(bool_or)

            for fluxc in df.keys():  # set it to 1 now, mask it out afterwards
                # print(df[fluxc].shape)
                df[fluxc][list_nan_points] = 1.0
            ######
            mydf_flattened = pd.DataFrame({key: df[key].reshape(matsize) for key in df.keys()})
        elif not input_is_array:
            mydf_flattened = pd.DataFrame({key: np.array([df[key]]) for key in df.keys()})
        else: # case in which is_array and matndim = 1:
            mydf_flattened = pd.DataFrame(df)
    elif isinstance(df, pd.DataFrame):
        mydf_flattened = df
    else:
        raise Exception("The type of df must be either dict or Pandas Data Frame!")

    #####
    model_loadname = '{}_model_{}_cosz_{}_adir_{}'.format(
        model, label, cosz, prefit_models_adir)

    with open(os.path.join(modeldir,
                '{}.pickle'.format(model_loadname)), 'rb') as pklf:
        loaded_model = pickle.load(pklf)

    # print(mydf_flattened)
    # print(mydf_flattened.shape)
    ypred = loaded_model.predict(mydf_flattened)



                     # RMC simulation were run at "prefit_models_adir" albedo.
    # adjust reflected fluxes to match desired albedom (frdir, frdif) or set to albedo=1 (frdirn, frdifn)
    # need to apply rescaling for reflected flux components
    if label in ['frdir', 'frdif']:
        ypred = ypred * (albedo / prefit_models_adir)
    if label in ['frdirn', 'frdifn']:
        ypred = ypred * (1.0 / prefit_models_adir)

    # nonlinear case: variable must be pre - transformed when fitting the model
    # print('label = {}, model = {}'.format(label, model))
    if label == 'fdir' and model=='NLR':
        # eta_pred = mlr_model.predict(Xrf_train)
        ymax = 2.5 / cosz
        ymin = -1.0
        eta_pred = loaded_model.predict(mydf_flattened)
        # eta_pred = ypred
        # ytilde_pred = np.exp(eta_pred) / (1.0 + np.exp(eta_pred))
        # ypred = (ymax - ymin) * ytilde_pred + ymin
        ypred = eta_pred
        # ypred = np.exp(eta_pred) - 1.0

        # ymlr_pred = np.zeros(np.shape(train_labels))
        # ytilde_pred = np.exp(ymlr_pred)/(1.0 + ymlr_pred)
        # ymlr_pred = ytilde_pred*(ymax-ymin) + ymin
        # neg_angle = Xrf_train['sian'].values < 0.0
        # neg_angle = mydf_flattened['sian'].values < 0.0
        # ypred[neg_angle] = -1.0

        # sunny area correction

        # print('does file {} exists?'.format(os.path.join(modeldir,
        #                 '{}.pickle'.format(model_loadname))))
        model_loadname = 'logistic_model_{}_cosz_{}_adir_{}'.format(
            label, cosz, prefit_models_adir)
        if os.path.exists(os.path.join(modeldir, '{}.pickle'.format(model_loadname))):

            # print('NLR: modify NLR to account for -1 areas')
            with open(os.path.join(modeldir,
                                   '{}.pickle'.format(model_loadname)), 'rb') as pklf:
                logistic_model = pickle.load(pklf)

            # logistic_hat = logistic_model.predict(mydf_flattened)
            logistic_hat_probs = logistic_model.predict_proba(mydf_flattened)[:, 1] # second value = bool 1
            # print('ypred ave before shade correction = {}'.format(ypred))
            ypred = ypred * logistic_hat_probs - 1.0 * (1 - logistic_hat_probs)
            # print('logistic average prob = {}'.format(np.mean(logistic_hat_probs)))
            # print('ypred ave after shade correction = {}'.format(ypred))

        # finally, multiply the ypred by the probability in [0,1] predicted by logistic model

    # if label == 'fdir' and np.min(ypred) < -1:
    #     print(ypred)

    # further postprocessing: set lower threshold for direct and diffuse fluxes
    if label in ['fdir', 'fdif']:
        # if label in ['fdir', 'fdif'] and model != 'NLR':
        ypred = np.maximum(-1.0, ypred)  # for all flux compts here
        # maxval_set = 8.5
        # ypred = np.minimum(ypred, maxval_set)  # for all flux compts here
        # print('WARNING: added upper thresh at {}'.format(maxval_set))
        # print('WARNING: no thresh at -1')

    if matndim > 1:
        ypred_original_shape = ypred.reshape(matshape)
        ypred_original_shape[list_nan_points] = np.nan #####
        # print('list of nan points:')
        # print(list_nan_points)
    elif input_is_scalar and not np.isscalar(ypred):
        assert np.size(ypred) == 1
        ypred_original_shape = ypred[0]
    else:
        # print('output df shape = ', ypred)
        ypred_original_shape = ypred
    # if matndim > 1: # case of 2D maps - images which may have missing data:
    return ypred_original_shape


def train_test_models(dfflux=None, dfvars=None, LABELS=None,
                      MODELS = None, GOF_METRICS = None,
                      mycosz=None,
                      myadir=None,
                      rf_n_estimators = 20,
                      rf_max_depth=8,
                      test_only = True, make_plot = True,
                      prefit_models_adir=None,
                      specific_predictors = None,
                      modeldir=None, testdir=None):

   """--------------------------------------------------------------------------
   Train (optional) and test predictive models
   multiple linear regression (mlr), random forest (rfr) and nnet (to be done)

   INPUT::
   dfflux -> labels
   dfvars -> predictors
   LABELS -> label keys
   mycosz -> cosine of zenith angle
   myadir -> land (direct) albedo
   myaveblock -> averaging block size
   test_only -> if True, use models already trained
   make_plot -> if True, save a scatter plot for model checking
   modeldir -> directory to store trained models
   testdir  -> directory to store plots / results for cross valid. if test_only

   OUTPUT:: saves models and (optionally) testing plots

   Train and test predictive models
   --------------------------------------------------------------------------"""
   # do_rf = False
   # nmodels = 3
   nfluxvars = len(LABELS)
   # R2 = np.zeros((nmodels, nfluxvars))*np.nan

   # metric_names = ['R2', 'MSE', 'MAE']
   # model_names = ['mlr', 'rfr', 'lee']
   # nmodels = len(model_names)
   metric_keys = ["{}_{}".format(a,b) for a,b in product(GOF_METRICS, MODELS)]
   metrics = {mykey:np.zeros(nfluxvars)*np.nan for mykey in metric_keys}


   for il, label in enumerate(LABELS):

       dfvars2 = dfvars.copy()
       # --------------------------------------------------------------
       # change variables based on label and cosz values
       # if label in ['frdirn', 'frdifn'] or (mycosz > 0.99 and label == 'fdir'):
       #     _ = dfvars2.pop('sian')
       # else:
       #     dfvars = dfvars_gen
       # if mycosz > 0.99 and label == 'fdir':
       #     print('Removing SIA from predictors in case cosz = 1')
       #     _ = dfvars.pop('sian')
       # --------------------------------------------------------------
       # ------------ divide data in training and testing -------------
       # print("preparing df for training and testing models")
       # do not do train-test separation here - data are spatially correlated anyway
       # better to use completely different domains for training and testing
       # and call this function is "test_only" mode after fitting the models

       # to do: generalize code for [user-defined] other sets of metrics
       assert 'R2' in GOF_METRICS
       # assert 'MAE' in GOF_METRICS
       # assert 'MSE' in GOF_METRICS
       # assert 'EVAR' in GOF_METRICS
       # assert 'MAXE' in GOF_METRICS
       # assert 'MAPE' in GOF_METRICS
       # dfred = {var: dfvars2[var] for var in dfvars2.keys()}
       dfred = {var: dfvars2[var] for var in dfvars2.keys() if var in specific_predictors[label]}
       dfred[label] = dfflux[label]  # add current label to the dict/df
       dfred = pd.DataFrame(dfred)
       # train_dataset = dfred
       # test_dataset = dfred
       train_features = dfred.copy()
       train_labels = train_features.pop(label)
       # test_features = train_features
       # test_labels = train_labels
       # scal = StandardScaler()
       # Xrf_train = scal.fit_transform(train_features)
       Xrf_train = train_features


       # train_labels = np.array(train_labels)
       # Xrf_train = np.array(train_features)

       # UPDATED - JUST READ DATAFRAMES::
       # train_labels = dfflux[label]
       # Xrf_train = dfvars



       if not test_only: # train models
           mlr_savename = 'MLR_model_{}_cosz_{}_adir_{}'.format(
                   label, mycosz, myadir)
           rfr_savename = 'RFR_model_{}_cosz_{}_adir_{}'.format(
                   label, mycosz, myadir)
           nlr_savename = 'NLR_model_{}_cosz_{}_adir_{}'.format(
               label, mycosz, myadir)
           logr_savename = 'logistic_model_{}_cosz_{}_adir_{}'.format(
               label, mycosz, myadir)

           # ------------------ fit NEURAL NETWORK MODEL ------------------
           # print('Training neural network model')
           # dnn_savedir = os.path.join(datadir, 'dnn_models')
           # dnn_savename = 'dnn_model_{}_cosz_{}_adir_{}'.format(
           #                         label, mycosz,  myadir)
           # dnn_model = dem.train_neuralnet(dfred, label=label,
           #                                 train_frac = train_frac,
           #                                 activation='linear',
           #                                 plot=False,
           #                                 save = True,
           #                                 savedir= dnn_savedir,
           #                                 savename=dnn_savename)
           # --------------------------------------------------------------


           # ------------- fit RANDOM FOREST MODEL ------------------------
           if 'RFR' in MODELS:
               # print('Training random forest model')
               # Xrf_test = scal.transform(test_features)
               rfr_model = RandomForestRegressor(
                            n_estimators=rf_n_estimators, max_depth=rf_max_depth)
               # print(rfr_model)
               # print(label)
               # print(Xrf_train.shape)
               # print(train_labels.shape)
               # print(np.sum(np.isnan(train_labels)))
               # print(np.sum(np.isnan(Xrf_train)))
               #
               # print(np.sum(~np.isfinite(train_labels)))
               # print(np.sum(~np.isfinite(Xrf_train)))
               #
               # print(np.mean(train_labels))
               # print(np.mean(Xrf_train))
               # print(np.min(train_labels))
               # print(np.min(Xrf_train))
               # print(np.max(train_labels))
               # print(np.max(Xrf_train))
               # AA = np.arange(24).reshape(6,4).astype(float)
               # BB = np.arange(6).astype(float)
               rfr_model.fit(Xrf_train, train_labels)
               # yrf_pred = myregressor.predict(Xrf_test)
               # yrf_pred_testalbedo = regressor.predict(Xrf_test_testalbedo)
               # get feature importances
               importance = rfr_model.feature_importances_
               # summarize feature importance
               # print('label => {}'.format(label))
               # feats = list(train_features.keys())
               feats = list(Xrf_train.keys())
               # print('features => {}'.format(feats))
               # for i, v in enumerate(importance):
                   # print('Feature %s: %0d, Score: %.5f' % (feats[i], i, v))
               with open(os.path.join(modeldir,
                       '{}.pickle'.format(rfr_savename)), 'wb') as pklf:
                   pickle.dump(rfr_model, pklf)
               # with open(os.path.join(modeldir,
               #         '{}.pickle'.format(rfr_savename)), 'rb') as pklf:
               #     rfr_model = pickle.load(pklf)

               # in this case repeat analysis varying RF configuration



           # --------------- fit MULTIPLE LINEAR REGRESSION ---------------
           if 'NLR' in MODELS:
               # for now, only dierct flux has NLR model
               # for other components, this will be equal to MLR model
               if not label == 'fdir':
                   nlr_model0 = LinearRegression()
                   nlr_model0.fit(Xrf_train, train_labels)
               else:
                   # add_nonlinear_link = True
                   # if add_nonlinear_link:
                   ## add nonlinear link value
                   # if label == 'fdir':
                   ymax = 2.5/mycosz
                   ymin = -1.0
                   ytilde = (train_labels.values - ymin)/(ymax - ymin) # normalized variable between [0,1]
                   pos_vals = np.logical_and( ytilde > 0, ytilde < 1)
                   is_sunny = (train_labels.values > -0.999).astype(int) # 1 == True == sunny
                   if np.min(is_sunny) == 1:
                       all_sunny = True
                       # print("no shades found")
                   else:
                       all_sunny = False

                   # print(is_sunny)
                   y_reduced = train_labels.values[is_sunny.astype(bool)].copy()
                   ytilde = ytilde[pos_vals]
                   # train_eta = np.log(ytilde/(1-ytilde))

                   # train_eta = np.log( y_reduced + 1.0)

                   # train_labels = eta
                   # pos_vals = np.logical_and( eta > 0, eta < 1)
                   # train_labels = eta[pos_vals]
                   # Xrf_train_reduced = Xrf_train.loc[pos_vals,:].copy()
                   # Xrf_train_reduced = Xrf_train.loc[is_sunny,:].copy()
                   # print(Xrf_train.shape)
                   # print(is_sunny.shape)
                   # print(y_reduced.shape)
                   Xrf_train_reduced = Xrf_train[is_sunny.astype(bool)].copy()
                   # print(Xrf_train_reduced.shape)

                   # print('adding nonlinear link.')
                   # print(np.shape(Xrf_train))
                   # print(np.shape(train_labels))
                   # print(np.min(train_labels))
                   # print(np.max(train_labels))
                   # print('Training nonlinear regression - fdir only')
                   nlr_model0 = LinearRegression()
                   # nlr_model0.fit(Xrf_train_reduced, train_eta)
                   nlr_model0.fit(Xrf_train_reduced, y_reduced)

                   if not all_sunny:
                       # print('Training logistic regression for shade prob. - fdir only')
                       scaler = StandardScaler()
                       lr = LogisticRegression()
                       logistic_model0 = Pipeline([('standardize', scaler),
                                          ('log_reg', lr)])
                       logistic_model0.fit(Xrf_train, is_sunny) # predict values where 1 = fdir > -1

                       with open(os.path.join(modeldir,
                                              '{}.pickle'.format(logr_savename)), 'wb') as pklf:
                           pickle.dump(logistic_model0, pklf)


               with open(os.path.join(modeldir,
                                      '{}.pickle'.format(nlr_savename)), 'wb') as pklf:
                   pickle.dump(nlr_model0, pklf)




           if 'MLR' in MODELS:
               # train_eta = train_labels.copy()
               # Xrf_train_reduced = Xrf_train.copy()
               # print('Training multiple linear regression')
               mlr_model0 = LinearRegression()
               mlr_model0.fit(Xrf_train, train_labels)
               # get variable importance
               # NOTE - THIS ASSUMES THAT VARIABLES HAVE BEEN NORMALIZED BEFORE
               # importance = mlr_model0.coef_
               # for i, v in enumerate(importance):
                   # print('Feature: %0d, Score: %.5f' % (i, v))
               with open(os.path.join(modeldir,
                           '{}.pickle'.format(mlr_savename)), 'wb') as pklf:
                   pickle.dump(mlr_model0, pklf)
               # with open(os.path.join(modeldir,
               #             '{}.pickle'.format(mlr_savename)), 'rb') as pklf:
               #     mlr_model = pickle.load(pklf)
               # -------------------------------------------------------------------


       # else: # prediction only using previously trained models
           # pass # load saved model from the function now
           # mlr_loadname = 'mlr_model_{}_cosz_{}_adir_{}'.format(
           #     label, mycosz, prefit_models_adir)
           # rfr_loadname = 'rfr_model_{}_cosz_{}_adir_{}'.format(
           #     label, mycosz, prefit_models_adir)
           # # read saved models and do predictions
           # if 'MLR' in MODELS:
           #     with open(os.path.join(modeldir,
           #              '{}.pickle'.format(mlr_loadname)), 'rb') as pklf:
           #         mlr_model = pickle.load(pklf)
           # if 'RFR' in MODELS:
           #     with open(os.path.join(modeldir,
           #             '{}.pickle'.format(rfr_loadname)), 'rb') as pklf:
           #         rfr_model = pickle.load(pklf)

       # LOAD SAVED MODELS AND PREDICT
       if not test_only:
           model_adir = myadir # use current adir if training now
       else:
           model_adir = prefit_models_adir # use adir of other trained model - potentially different
       # now both in case of train or test, read saved models and predict y
       ypred_dict = {}
       for model in MODELS:
           ypred_dict[model] = model_predict_interp(Xrf_train, label=label, cosz=mycosz, albedo=myadir,
                                model=model, modeldir=modeldir, prefit_models_adir=model_adir)

           for metric in GOF_METRICS:
               metrics['{}_{}'.format(metric, model)][il] = compute_gof_metrics(
                                train_labels, ypred_dict[model], metric=metric)

       # # do metrics one by one here
       # if 'MLR' in MODELS:
       #     ymlr_pred = my_model_predict()
       #     # ymlr_pred = mlr_model.predict(Xrf_train)
       #     # ymlr_pred = np.maximum(-1.0, ymlr_pred)  # for all flux compts here
       #
       # if 'NLR' in MODELS:
       #     if add_nonlinear_link and label == 'fdir':
       #         eta_pred = mlr_model.predict(Xrf_train)
       #         ytilde_pred = np.exp(eta_pred)/(1.0 + np.exp(eta_pred))
       #         ymlr_pred = (ymax - ymin)*ytilde_pred + ymin
       #         # ymlr_pred = np.zeros(np.shape(train_labels))
       #         # ytilde_pred = np.exp(ymlr_pred)/(1.0 + ymlr_pred)
       #         # ymlr_pred = ytilde_pred*(ymax-ymin) + ymin
       #         neg_angle = Xrf_train['sian'].values < 0.0
       #         ymlr_pred[neg_angle] = -1.0
       #     else:
       #         ymlr_pred = mlr_model.predict(Xrf_train)
       #         ymlr_pred = np.maximum(-1.0, ymlr_pred) # for all flux compts here
       #
       #     metrics['R2_MLR'][il] = r2_score(train_labels, ymlr_pred)
       #     metrics['MAE_MLR'][il] = mean_absolute_error(train_labels, ymlr_pred)
       #     metrics['MSE_MLR'][il] = mean_squared_error(train_labels, ymlr_pred)
       #     metrics['EVAR_MLR'][il] = explained_variance_score(train_labels, ymlr_pred)
       #     metrics['MAXE_MLR'][il] = max_error(train_labels, ymlr_pred)
       #     metrics['MAPE_MLR'][il] = mean_absolute_percentage_error(train_labels, ymlr_pred)
       #
       # if 'RFR' in MODELS:
       #     yrfr_pred = rfr_model.predict(Xrf_train)
       #     yrfr_pred = np.maximum(-1.0, yrfr_pred) # for all flux compts here
       #     metrics['R2_RFR'][il] = r2_score(train_labels, yrfr_pred)
       #     metrics['MAE_RFR'][il] = mean_absolute_error(train_labels, yrfr_pred)
       #     metrics['MSE_RFR'][il] = mean_squared_error(train_labels, yrfr_pred)
       #     metrics['EVAR_RFR'][il] = explained_variance_score(train_labels, yrfr_pred)
       #     metrics['MAXE_RFR'][il] = max_error(train_labels, yrfr_pred)
       #     metrics['MAPE_RFR'][il] = mean_absolute_percentage_error(train_labels, yrfr_pred)
       #
       #
       #
       # if label not in ['fcoup', 'fcoupn'] and 'LEE' in MODELS:
       #     ylee_pred = lee_model_predict(train_dataset,
       #                                   label=label, cosz=mycosz,
       #                                   albedo=myadir)
       #     metrics['R2_LEE'][il] = r2_score(train_labels, ylee_pred)
       #     metrics['MAE_LEE'][il] = mean_absolute_error(train_labels, ylee_pred)
       #     metrics['MSE_LEE'][il] = mean_squared_error(train_labels, ylee_pred)
       #     metrics['EVAR_LEE'][il] = explained_variance_score(train_labels, ylee_pred)
       #     metrics['MAXE_LEE'][il] = max_error(train_labels, ylee_pred)
       #     metrics['MAPE_LEE'][il] = mean_absolute_percentage_error(train_labels, ylee_pred)

       if make_plot:
           ave_obs = np.mean(train_labels)
           if test_only:
               typetitle = 'cross val'
           else:
               typetitle = 'same sample'

           plt.figure(figsize=(10, 10))
           if label not in ['fcoup', 'fcoupn'] and 'LEE' in MODELS:
               # plt.plot(train_labels, ylee_pred, 'or',
               ave_pred = np.mean(ypred_dict['LEE'])
               plt.plot(train_labels, ypred_dict['LEE'], 'or',
                        label=r'LEE $R^2 = {:.2f}$, $\langle obs \rangle = {:.3f}$, $\langle pred \rangle = {:.3f}$'.format(
                        metrics['R2_LEE'][il], ave_obs, ave_pred))
           if 'MLR' in MODELS:
               # plt.plot(train_labels, ymlr_pred, 'ob',
               ave_pred = np.mean(ypred_dict['MLR'])
               plt.plot(train_labels, ypred_dict['MLR'], 'ob',
                        label=r'MLR $R^2 = {:.2f}$, $\langle obs \rangle = {:.3f}$, $\langle pred \rangle = {:.3f}$'.format(
                        metrics['R2_MLR'][il], ave_obs, ave_pred))
               # label=R'MLR $r^2 = {:.2f}$'.format(metrics['R2_MLR'][il]))
           if 'RFR' in MODELS:
               ave_pred = np.mean(ypred_dict['RFR'])
               # plt.plot(train_labels, yrfr_pred, 'og',
               plt.plot(train_labels, ypred_dict['RFR'], 'og',
                        label=r'RFR $R^2 = {:.2f}$, $\langle obs \rangle = {:.3f}$, $\langle pred \rangle = {:.3f}$'.format(
                        metrics['R2_RFR'][il], ave_obs, ave_pred))
                        # label=r'RFR $R^2 = {:.2f}$'.format(metrics['R2_RFR'][il]))

           if 'NLR' in MODELS:
               ave_pred = np.mean(ypred_dict['NLR'])
               # plt.plot(train_labels, yrfr_pred, 'og',
               plt.plot(train_labels, ypred_dict['NLR'], '*c',
                        label=r'NLR $R^2 = {:.2f}$, $\langle obs \rangle = {:.3f}$, $\langle pred \rangle = {:.3f}$'.format(
                        metrics['R2_NLR'][il], ave_obs, ave_pred))
                        # label=r'NLR $R^2 = {:.2f}$'.format(metrics['R2_NLR'][il]))
           plt.plot(train_labels, train_labels, 'k')
           plt.title('{} pred {} flux, cosz = {}.'.format(typetitle, label, mycosz))
           plt.xlabel('{} - Monte Carlo Simulation'.format(label))
           plt.ylabel('{} - model prediction'.format(label))
           lee_savename = '{}_model_{}_cosz_{}_adir_{}.png'.format(
               typetitle, label, mycosz, myadir)
           plt.legend(loc='upper left')
           if not test_only:
               modeldir_fig = os.path.join(modeldir, "gof_training_plots")
               os.system("mkdir -p {}".format(modeldir_fig))
               plt.savefig(os.path.join(modeldir_fig, lee_savename))
           else:
               testdir_fig = os.path.join(testdir, "gof_testing_plots")
               os.system("mkdir -p {}".format(testdir_fig))
               plt.savefig(os.path.join(testdir_fig, lee_savename))
           plt.close()
   return metrics


def compute_gof_metrics(train_labels, pred_labels, metric='R2'):
    if metric == 'R2':
        my_metric = r2_score(train_labels, pred_labels)
    elif metric == 'MAE':
        my_metric = mean_absolute_error(train_labels, pred_labels)
    elif metric == 'MSE':
        my_metric = mean_squared_error(train_labels, pred_labels)
    elif metric == 'EVAR':
        my_metric = explained_variance_score(train_labels, pred_labels)
    elif metric == 'MAXE':
        my_metric = max_error(train_labels, pred_labels)
    elif metric == 'MAPE':
        my_metric = mean_absolute_percentage_error(train_labels, pred_labels)
    return my_metric


def init_metrics_dataset(DOMAINS, AVEBLOCKS, CROP_BUFFERS, ADIRs,
                         LABELS, COSZs,MODELS, GOF_METRICS,
                         RF_N_ESTIMATORS, RF_MAX_DEPTH):
    # initialize a dataset to store model result
    # note: cosz and adir (floats) must be indexed using their integer index
    ncosz = len(COSZs)
    nlabels = len(LABELS)
    nadir = len(ADIRs)
    nbuffers = len(CROP_BUFFERS)
    naveblocks = len(AVEBLOCKS)
    ndomains = len(DOMAINS)
    nmodels = len(MODELS)
    nmetrics = len(GOF_METRICS)
    number_n_estimators = len(RF_N_ESTIMATORS)
    number_max_depth = len(RF_MAX_DEPTH)

    modeldata = xr.Dataset(
        {
            "GOFs": (
                ("domain", "buffer", "aveblock", "iadir", "icosz", "model",
                 "label", "metric", "rf_n_estimators", "rf_max_depth"),
                np.zeros((ndomains,  nbuffers, naveblocks,
                          nadir, ncosz, nmodels, nlabels, nmetrics,
                          number_n_estimators, number_max_depth),
                         dtype=np.float32)),

                "ADIRs": ("iadir", ADIRs),
                "COSZs": ("icosz", COSZs)
        },
        coords={
            "model": MODELS,
            "label": LABELS,
            "metric": GOF_METRICS,
            "domain": DOMAINS,
            "buffer": CROP_BUFFERS,
            "aveblock": AVEBLOCKS,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "rf_max_depth": RF_MAX_DEPTH,
            "icosz": np.arange(ncosz), # integer cosz index
            "iadir": np.arange(nadir)  # integer cosz index
        },
        attrs={
            "test_only": 0, #
            "cv_training_domain": 'None', # fill only for cross-validation
            "cv_training_aveblock": 'None'  # fill only for cross-validation
        }
    )
    return modeldata


def read_PP_3D_differences(datadir=None, outdir=None, do_average=True,
                           aveblock = 110, crop_buffer = 0.25,
                           do_plots = False, outfigdir = None,
                           save_data = True,
                           FTOAnorm=1361.0, do_debug=False):
    """-------------------------------------------------------------------------
    Read the output of a Radiation Monte Carlo (RMC) simulation
    compute differences between 3D and PP simulations
    averaging results over rectangular tiles of size "aveblock" grid cells
    ( e.g., for 90m SRTM data the output resolution will be "aveblock" * 90m
    cropping a fraction "crop_buffer" of the simulation domain
    (e.g. for crop_buffer=0.25 only the central half domain is kept
    and a 0.25 fraction of each side is discarded to avoid boundary effects)
    Fluxes are normalized with respect to TOA value "FTOAnorm" provided [W m^-2]

    INPUT::
    datadir     -> input folder with results of RMC simulation to read
    outdir      -> output folder to store netcdf with results
    do_average  -> if True, average results over tiles of size aveblock time
                   the original simulation terrain resolution
    aveblock    -> averaging window size, multiple of original simulation res.
    crop_buffer -> domain fraction to discard at each boundary
    FTOAnorm    -> TOP of atmosphere irradiance to get dimensional fluxes
    do_debug    -> plot and show diagnostic plots
    do_plots    -> produce quality plots of terrain variables and fluxes
    outfigdir   -> output folder to store __do_plots__figures
    save_data   -> only if True write netcdf with results (mostly for plotting)

    OUTPUT::
    write 2 netcdf files with terrain variables and difference between 3D and PP
    simulations. One contains separate results for each simulation albedo
    values, the other contains a result averaged over all albedo values.
    If do_plots, store figures in outfigdir
    -------------------------------------------------------------------------"""

    # read the list of jobs for the given RMC simulation:
    df3d = pd.read_csv( os.path.join(datadir, 'list_sim_cases_3D.csv'))
    # dfpp = pd.read_csv( os.path.join(datadir, 'list_sim_cases_PP.csv'))

    COSZs = list(np.unique(df3d['cosz']))
    PHIs = list(np.unique(df3d['phi']))
    ADIRs = list(np.unique(df3d['adir']))
    # print('cosz available = {}'.format(COSZs))
    # print('phi  available = {}'.format(PHIs))
    # print('adir available = {}'.format(ADIRs))

    # fluxesc = ['fdir', 'fdif', 'frdir', 'frdif', 'fcoup']

    nazimuths = np.size(PHIs)
    nadir = np.size(ADIRs)
    ncosz = len(COSZs)
    # adir_prod = np.prod(ADIRs)

    rest_ip = load_static_terrain_vars(
                                buffer=crop_buffer,
                                do_average=do_average,
                                aveblock=aveblock,
                                datadir=datadir)
    # rest_ip = rest_ip_static.copy()

    # loop on cosz, albedos and azimuth values to extract simulation results
    for ic, mycosz in enumerate(COSZs):
        for ia, myadir in enumerate(ADIRs):

            # print('reading RMC simulation results '
            #       'for cosz = {}, adir = {}'.format(mycosz ,myadir))

            mydf3d = df3d[ (df3d['cosz']== mycosz) & (df3d['adir'] == myadir)]
            assert mydf3d.shape[0] == nazimuths

            # read fluxes over plane-parallel domain
            # We repeat this even though it is the same for each phi values
            # it only outputs scalar values, so it is not memory intensive
            # move it out the loop - it would be cleaner
            respp_ip = load_pp_fluxes(cosz=mycosz,
                                      adir=myadir,
                                      FTOAnorm=FTOAnorm,
                                      do_average=False, aveblock=1,
                                      buffer=crop_buffer,
                                      datadir=datadir)

            # for ip, myphi in enumerate(PHIs):
            for ip, myphi in enumerate(PHIs):

                # read terrain variables - only once
                # it computes SIA, so it must stay within the COSZ and PHI loops
                # rest_ip = load_terrain_vars(cosz=mycosz, phi=myphi,
                #                              buffer=crop_buffer,
                #                              do_average=do_average,
                #                              aveblock=aveblock,
                #                              datadir= datadir)
                rest_ip_dynamic = comp_solar_incidence_field(rest_ip, cosz=mycosz, phi=myphi,
                            buffer = crop_buffer,do_average = do_average, aveblock = aveblock)
                # print([np.mean(rest_ip[a]) for a in rest_ip.keys()])
                rest_ip.update(rest_ip_dynamic)

                # read fluxes over 3d terrain
                res3d_ip = load_3d_fluxes(FTOAnorm=FTOAnorm,
                                           cosz = mycosz,
                                           phi = myphi,
                                           adir = myadir,
                                           do_average=do_average,
                                           aveblock=aveblock,
                                           buffer=crop_buffer,
                                           datadir = datadir)

                # # read fluxes over plane-parallel domain
                # # We repeat this even though it is the same for each phi values
                # # it only outputs scalar values, so it is not memory intensive
                # # move it out the loop - it would be cleaner
                # respp_ip = load_pp_fluxes(cosz=mycosz,
                #                           adir=myadir,
                #                           FTOAnorm=FTOAnorm,
                #                           do_average=False, aveblock=1,
                #                           buffer=crop_buffer,
                #                           datadir=datadir)

                # res3d_ip['FMcoup'] = corrected_fcoup.copy()

                ###### correct reflected and direct fluxes (normalize by albedo::
                # (as of now, already done in the functions above!) #
                ###### correct the TCF, SVF dividing by cos(slope) -> done!
                ###### correct SIA dividing by cos(slope) and cosz -> done!


                x = rest_ip['x']
                y = rest_ip['y']
                # invy = rest_ip['invy']
                # Z = rest_ip['Z']

                # load predictors (terrain variables)::
                svfn_field =  rest_ip['SVFnorm']
                tcfn_field =  rest_ip['TCFnorm']
                sian_field =  rest_ip['SIAnorm']
                sde_field =  rest_ip['sde']
                ele_field =  rest_ip['Z']
                elen_field =  rest_ip['elen']

                # sianz_field =  rest_ip['SIAnormz']
                sia0_field =  rest_ip['SIA0']

                svfn =    np.ravel( svfn_field   )
                tcfn =    np.ravel( tcfn_field   )
                sian =    np.ravel( sian_field   )
                sde =    np.ravel( sde_field     )
                ele =    np.ravel( ele_field     )
                elen =    np.ravel( elen_field     )


                sia0 =    np.ravel( sia0_field   )
                # sianz =    np.ravel( sianz_field   )

                if do_debug:
                    # matplotlib.use('Qt5Agg') # dyn show plots
                    plt.figure()
                    plt.imshow(res3d_ip['FMrdir'])
                    # plt.imshow(rest_ip['Z'])
                    plt.show()

                    plt.figure()
                    plt.hist(np.ravel(res3d_ip['FMrdir']), bins=60)
                    plt.show()
                    print('showing diagnostic plot')

                # read the original terrain configuration factor, not normalized
                # by cos(slope), as may be needed to normalize coupled flux
                tcf0 =    np.ravel( rest_ip['TCF0']   )
                svf0 =    np.ravel( rest_ip['SVF0']   )

                # read labels (3D - PP flux differences)::
                # compute (and flatten) matrices of differences between 3D and PP fluxes

                fFMdir_field =  ( res3d_ip['FMdir'] -  respp_ip['Fdir_pp'])/respp_ip['Fdir_pp']
                fFMdif_field =  ( res3d_ip['FMdif'] -  respp_ip['Fdif_pp'])/respp_ip['Fdif_pp']
                fFMrdir_field = ( res3d_ip['FMrdir'])/ respp_ip['Fdir_pp']
                fFMrdif_field = ( res3d_ip['FMrdif'])/ respp_ip['Fdif_pp']
                fFMcoup_field = ( res3d_ip['FMcoup'] - respp_ip['Fcoup_pp'])/respp_ip['Fcoup_pp']

                fFMdir =  np.ravel( fFMdir_field)
                fFMdif =  np.ravel( fFMdif_field)
                fFMrdir = np.ravel( fFMrdir_field)
                fFMrdif = np.ravel( fFMrdif_field)
                fFMcoup = np.ravel( fFMcoup_field)

                if do_plots and ip==0 and ia==nadir-1:
                    plot_terrain_fields(x, y,
                            ele_field, svfn_field,
                            tcfn_field, sian_field, elen_field,
                            fFMdir_field,
                            fFMdif_field, fFMrdir_field,
                            fFMrdif_field, fFMcoup_field,
                            outfigdir=outfigdir, cosz=mycosz,
                            adir=myadir, phi = myphi,
                            aveblock=aveblock)


                # plane parallel ratio of coupled to (direct + diffuse fluxes)
                coup_frac_3d = res3d_ip['Fcoup'] / (res3d_ip['Fdir'] + res3d_ip['Fdif'])
                coup_frac_pp = respp_ip['Fcoup_pp'] / (respp_ip['Fdir_pp'] + respp_ip['Fdif_pp'])

                # normalized single and multiple reflection flux components
                # reflected fluxes -> simply divide by simulation albedos
                # coupled flux -> skip normalization for now,
                #                 will probably need a more complex fix later
                fFMcoupN = fFMcoup
                fFMrdirN = fFMrdir / myadir
                fFMrdifN = fFMrdif / myadir

                # use a single albedo value for now
                # initialize arrays with results only once for each cosz value
                if ip == 0 and ia == 0: # in this case initialize result arrays

                    npix = np.size(svfn)
                    mysize = npix * len(PHIs)
                    nadirs = len(ADIRs)

                    fDIR = np.zeros(  (mysize, nadirs) )
                    fDIF = np.zeros(  (mysize, nadirs) )
                    fRDIR = np.zeros( (mysize, nadirs) )
                    fRDIF = np.zeros( (mysize, nadirs) )
                    fCOUP = np.zeros( (mysize, nadirs) )

                    fRDIRN = np.zeros( (mysize, nadirs) )
                    fRDIFN = np.zeros( (mysize, nadirs) )
                    fCOUPN = np.zeros( (mysize, nadirs) )

                    SIAN = np.zeros(mysize)
                    SVFN = np.zeros(mysize)
                    TCFN = np.zeros(mysize)
                    TCF0 = np.zeros(mysize)
                    SVF0 = np.zeros(mysize)
                    SDE = np.zeros(mysize)
                    ELE = np.zeros(mysize)
                    ELEN = np.zeros(mysize)
                    SIA0 = np.zeros(mysize)

                    # intialize xarray dataset(s) to store results (only once!)
                    if ic == 0:
                        ds = xr.Dataset(
                                {
                                "fdir": (("npix",  "icosz"),  np.zeros((mysize, ncosz), dtype=np.float32)),
                                "fdif": (("npix",  "icosz"),  np.zeros((mysize, ncosz), dtype=np.float32)),
                                "frdir": (("npix", "icosz"), np.zeros((mysize, ncosz), dtype=np.float32)),
                                "frdif": (("npix", "icosz"), np.zeros((mysize, ncosz), dtype=np.float32)),
                                "fcoup": (("npix", "icosz"), np.zeros((mysize, ncosz), dtype=np.float32)),
                                    "frdirn": (("npix", "icosz"), np.zeros((mysize, ncosz), dtype=np.float32)),
                                    "frdifn": (("npix", "icosz"), np.zeros((mysize, ncosz), dtype=np.float32)),
                                    "fcoupn": (("npix", "icosz"), np.zeros((mysize, ncosz), dtype=np.float32)),
                                    "sian": (("npix",   "icosz"),   np.zeros((mysize, ncosz), dtype=np.float32)),
                                    "sia0": (("npix",   "icosz"),   np.zeros((mysize, ncosz), dtype=np.float32)),
                                "svfn": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "tcfn": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "tcf0": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "svf0": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "sde": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "ele": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "elen": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "COSZs": ("icosz",  COSZs)
                                },
                                coords={
                                    "npix": np.arange(mysize),
                                    "icosz": np.arange(ncosz)
                                }
                                # attrs={
                                #     "coup_frac_pp": coup_frac_pp
                                # }
                            )

                        ds_ia = xr.Dataset(
                            {
                                "fdir": (("npix",  "icosz", "iadir"),  np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "fdif": (("npix",  "icosz", "iadir"),  np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "frdir": (("npix", "icosz", "iadir"), np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "frdif": (("npix", "icosz", "iadir"), np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "fcoup": (("npix", "icosz", "iadir"), np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "frdirn": (("npix", "icosz", "iadir"), np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "frdifn": (("npix", "icosz", "iadir"), np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "fcoupn": (("npix", "icosz", "iadir"), np.zeros((mysize, ncosz, nadir), dtype=np.float32)),
                                "sian": (("npix",   "icosz"),   np.zeros((mysize, ncosz), dtype=np.float32)),
                                "sia0": (("npix",   "icosz"),   np.zeros((mysize, ncosz), dtype=np.float32)),
                                "svfn": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "tcfn": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "tcf0": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "svf0": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "sde": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "ele": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "elen": (("npix"),   np.zeros((mysize), dtype=np.float32)),
                                "COSZs": ("icosz", COSZs),
                                "ADIRs": ("iadir", ADIRs),
                                "coup_frac_pp": (("icosz","iadir"), np.zeros((ncosz, nadir), dtype=np.float32)),
                                "coup_frac_3d": (("icosz","iadir"), np.zeros((ncosz, nadir), dtype=np.float32))
                            },
                            coords={
                                "npix": np.arange(mysize),
                                "icosz": np.arange(ncosz),
                                "iadir": np.arange(nadir)
                            }
                            # attrs={
                            #     "coup_frac_pp": coup_frac_pp
                            # }
                        )

                # for each phi and adir value, store results in a 2D array
                fDIR[npix*ip:npix*(ip+1) , ia] = fFMdir
                fDIF[npix*ip:npix*(ip+1) , ia] = fFMdif
                fRDIR[npix*ip:npix*(ip+1), ia] = fFMrdir
                fRDIF[npix*ip:npix*(ip+1), ia] = fFMrdif
                fCOUP[npix*ip:npix*(ip+1), ia] = fFMcoup

                fRDIRN[npix*ip:npix*(ip+1) , ia] = fFMrdirN
                fRDIFN[npix*ip:npix*(ip+1) , ia] = fFMrdifN
                fCOUPN[npix*ip:npix*(ip+1),  ia] = fFMcoupN

                if ia == 0: # terrain variables: read them only for the first adir
                    SIAN[npix*ip:npix*(ip+1) ] = sian
                    SIA0[npix*ip:npix*(ip+1) ] = sia0
                    SVFN[npix*ip:npix*(ip+1) ] = svfn
                    TCFN[npix*ip:npix*(ip+1) ] = tcfn
                    TCF0[npix*ip:npix*(ip+1) ] = tcf0
                    SVF0[npix*ip:npix*(ip+1) ] = svf0
                    SDE[npix*ip:npix*(ip+1) ] = sde
                    ELE[npix*ip:npix*(ip+1) ] = ele
                    ELEN[npix*ip:npix*(ip+1) ] = elen

            # save studd for current ia, myadir only:
                # initialize ds_ia with results for current myadir


            # Save results for different cosz in the xarray dataset
            ds_ia['fdir'][:, ic , ia] = fDIR[:,ia]
            ds_ia['fdif'][:, ic , ia] = fDIF[:,ia]
            ds_ia['frdir'][:, ic, ia] = fRDIR[:,ia]
            ds_ia['frdif'][:, ic, ia] = fRDIF[:,ia]
            ds_ia['fcoup'][:, ic, ia] = fCOUP[:,ia]
            ds_ia['frdirn'][:, ic, ia] = fRDIRN[:,ia]
            ds_ia['frdifn'][:, ic, ia] = fRDIFN[:,ia]
            ds_ia['fcoupn'][:, ic, ia] = fCOUPN[:,ia]
            ds_ia['sian'][:, ic] = SIAN # this changes with cosz and phi
            ds_ia['sia0'][:, ic] = SIA0 # this changes with cosz and phi
            ds_ia['coup_frac_pp'][ic, ia] = coup_frac_pp
            ds_ia['coup_frac_3d'][ic, ia] = coup_frac_3d
            if ic==0:
                ds_ia['sde'][:] = SDE
                ds_ia['svfn'][:] = SVFN
                ds_ia['tcfn'][:] = TCFN
                ds_ia['tcf0'][:] = TCF0
                ds_ia['svf0'][:] = SVF0
                ds_ia['ele'][:] = ELE
                ds_ia['elen'][:] = ELEN



        # average results over different adir values
        fDIR_aveadir = np.mean(fDIR, axis=1)
        fDIF_aveadir = np.mean(fDIF, axis=1)
        fRDIR_aveadir = np.mean(fRDIR, axis=1)
        fRDIF_aveadir = np.mean(fRDIF, axis=1)
        fCOUP_aveadir = np.mean(fCOUP, axis=1)


        fRDIRN_aveadir = np.mean(fRDIRN, axis=1)
        fRDIFN_aveadir = np.mean(fRDIFN, axis=1)
        fCOUPN_aveadir = np.mean(fCOUPN, axis=1)

        # Save results for different cosz in the xarray dataset
        ds['fdir'][:,ic] = fDIR_aveadir
        ds['fdif'][:,ic] = fDIF_aveadir
        ds['frdir'][:,ic] = fRDIR_aveadir
        ds['frdif'][:,ic] = fRDIF_aveadir
        ds['fcoup'][:,ic] = fCOUP_aveadir
        ds['frdirn'][:,ic] = fRDIRN_aveadir
        ds['frdifn'][:,ic] = fRDIFN_aveadir
        ds['fcoupn'][:,ic] = fCOUPN_aveadir
        ds['sian'][:,ic] = SIAN
        ds['sia0'][:,ic] = SIA0
        if ic == 0:
            ds['sde'][:] = SDE
            ds['svfn'][:] = SVFN
            ds['tcfn'][:] = TCFN
            ds['tcf0'][:] = TCF0
            ds['svf0'][:] = SVF0
            ds['ele'][:] = ELE
            ds['elen'][:] = ELEN

    # save result in a netcdf file
    if save_data:
        adir_prod = datadir.split('_')[-1]
        # print("***************************")
        # print(adir_prod)
        # print("***************************")
        file_res = os.path.join(outdir,
              'train_test_data_size_{}_adir_ave_{}.nc'.format(aveblock, adir_prod))
        file_res_ia = os.path.join(outdir,
              'train_test_data_size_{}_adir_singles_{}.nc'.format(
              aveblock, adir_prod))
        ds.to_netcdf(file_res)
        ds_ia.to_netcdf(file_res_ia)

    ds.close()
    ds_ia.close()

    return # we are done here


def plot_terrain_fields(x, y, ele_field, svfn_field,
                        tcfn_field, sian_field, elen_field,
                        fFMdir_field,
                        fFMdif_field, fFMrdir_field,
                        fFMrdif_field, fFMcoup_field,
                        outfigdir=None, cosz=None,
                        adir=None, phi=None,
                        aveblock=None):
    
    fluxes = {'dir':fFMdir_field, 'dif':fFMdif_field, 'rdir':fFMrdir_field, 'rdif':fFMrdif_field, 'coup':fFMcoup_field}
    fluxes_names = list(fluxes.keys())
    fluxes_symbols = [r'$f_{dir}$', r'$f_{dif}$', r'$f_{rdir}$', r'$f_{rdif}$', r'$f_{coup}$']
    nfluxes = len(fluxes_names)

    # preds = {'svfn':svfn_field, 'tcfn':tcfn_field, 'sian':sian_field}
    preds = {'svfn':svfn_field, 'tcfn':tcfn_field, 'sian':sian_field, 'elen':elen_field}
    preds_names = list(preds.keys())
    preds_symbols = [r'$\tilde{V}_{d}$', r'$\tilde{C}_{t}$', r'$\tilde{\mu}_{i}$', r"\tilde{h}"]
    npreds = len(preds_names)

    gridsz = 25
    figsize=5
    pad = 7


    ############################################################################
    read_dimensional_fluxes = False
    if read_dimensional_fluxes:
        ### read Fu-Liou fluxes
        ### and compute dimensional flux differences (W m^-2)
        ### do only for cosz = 0.5 and adir = 0.3 for now
        simul_folder = os.path.join('//', 'home', 'enrico', 'Documents',
                                    'dem_datasets')
        fulioudir = os.path.join(simul_folder, 'FuLiouresults_EastAlps_cosz05.csv')
        dfuliou = pd.read_csv(fulioudir)

        # use clear atmosphere for now
        FDIR = dfuliou['direct_alb03'][0]
        FDIF = dfuliou['diffuse_alb03'][0]
        FCOUP = dfuliou['coupled'][0]


        dim_fFMdir_field =  FDIR * fFMdir_field
        dim_fFMdif_field =  FDIF * fFMdif_field
        dim_fFMrdir_field = FDIR * fFMrdir_field
        dim_fFMrdif_field = FDIF * fFMrdif_field
        dim_fFMcoup_field = FCOUP* fFMcoup_field



    else:
        # pass
        dim_fFMdir_field =   fFMdir_field
        dim_fFMdif_field =   fFMdif_field
        dim_fFMrdir_field =  fFMrdir_field
        dim_fFMrdif_field =  fFMrdif_field
        dim_fFMcoup_field =  fFMcoup_field

    dimfluxes = {'dir': dim_fFMdir_field, 'dif': dim_fFMdif_field,
                 'rdir': dim_fFMrdir_field, 'rdif': dim_fFMrdif_field,
                 'coup': dim_fFMcoup_field}
    #

    ############################################################################


    #### Plot scatter plots all [dimless] predictors vs labels
    fig, axes = plt.subplots(ncols=npreds, nrows=nfluxes,
                             figsize=(figsize*npreds, figsize*nfluxes))
    countp = 0
    for j, fluxj in enumerate(fluxes_names):
        # print(j, fluxj)
        for i, predi in enumerate(preds_names):
            # print(i, predi)
            axes[j,i].scatter( np.ravel(preds[predi]), np.ravel(fluxes[fluxj]))
            if i==0:
                axes[j,i].annotate(fluxes_symbols[j],
                                   xy=(0, 0.5), xytext=(-axes[j,i].yaxis.labelpad - pad, 0),
                                   xycoords=axes[j,i].yaxis.label, textcoords='offset points',
                                   size='large', ha='right', va='center')
            if j == 0:
                axes[j,i].set_title('{}'.format(preds_symbols[i]))
            if j == nfluxes-1:
                axes[j,i].set_xlabel(r'{}'.format(preds_symbols[i]))

            axes[j,i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                           transform=axes[j,i].transAxes,
                           size=20, weight='bold')
            countp +=1
    plt.savefig(os.path.join(outfigdir,
                'scatter_predictors_cosz_{:0.2f}_ave_{:d}_adir_{:0.2f}_phi_{:0.2f}.png'.format(
                cosz, trunc(aveblock), adir, phi)), dpi = 300)
    plt.close()

    #### Plot hexbin-scatter plots all predictors vs labels [dimless fluxes]
    fig, axes = plt.subplots(ncols=npreds, nrows=nfluxes,
                             figsize=(figsize*npreds, figsize*nfluxes))
    countp = 0
    for j, fluxj in enumerate(fluxes_names):
        # print(j, fluxj)
        for i, predi in enumerate(preds_names):
            # print(i, predi)
            axes[j,i].hexbin( np.ravel(preds[predi]), np.ravel(fluxes[fluxj]),
                          cmap=plt.cm.Greens, gridsize = gridsz, bins = 'log')
            if i==0:
                axes[j,i].annotate(fluxes_symbols[j],
                    xy=(0, 0.5), xytext=(-axes[j,i].yaxis.labelpad - pad, 0),
                    xycoords=axes[j,i].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
            if j == 0:
                axes[j,i].set_title('{}'.format(preds_symbols[i]))
            if j == nfluxes-1:
                axes[j,i].set_xlabel(r'{}'.format(preds_symbols[i]))

            axes[j,i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                           transform=axes[j,i].transAxes,
                           size=20, weight='bold')
            countp +=1
    plt.savefig(os.path.join(outfigdir,
                'hexbin_predictors_cosz_{:0.2f}_ave_{:d}_adir_{:0.2f}_phi_{:0.2f}.png'.format(
                cosz, trunc(aveblock), adir, phi)), dpi = 300)
    plt.close()

    #### Plot scatter plots all predictors vs labels
    #### with DIMENSIONAL FLUX DIFFERENCES [W m^-2] using Fu-Liou flux values
    fig, axes = plt.subplots(ncols=npreds, nrows=nfluxes,
                             figsize=(figsize * npreds, figsize * nfluxes))
    countp = 0
    for j, fluxj in enumerate(fluxes_names):
        # print(j, fluxj)
        for i, predi in enumerate(preds_names):
            # print(i, predi)
            axes[j, i].hexbin(np.ravel(preds[predi]), np.ravel(dimfluxes[fluxj]),
                              cmap=plt.cm.Greens, gridsize=gridsz, bins='log')
            if i == 0:
                axes[j, i].annotate(fluxes_symbols[j],
                                    xy=(0, 0.5), xytext=(
                    -axes[j, i].yaxis.labelpad - pad, 0),
                                    xycoords=axes[j, i].yaxis.label,
                                    textcoords='offset points',
                                    size='large', ha='right', va='center')
            if j == 0:
                axes[j, i].set_title('{}'.format(preds_symbols[i]))
            if j == nfluxes - 1:
                axes[j, i].set_xlabel(r'{}'.format(preds_symbols[i]))

            axes[j, i].text(-0.1, 1.1, string.ascii_uppercase[countp],
                            transform=axes[j, i].transAxes,
                            size=20, weight='bold')
            countp += 1
    plt.savefig(os.path.join(outfigdir,
            'dimhexbin_predictors_cosz_{:0.2f}_ave_{:d}_adir_{:0.2f}_phi_{:0.2f}.png'.format(
            cosz, trunc(aveblock), adir, phi)), dpi=300)
    plt.close()

    #### Plot maps of labels - normalized flux differences
    shading = 'auto'
    myx = x/1000
    myy = y/1000
    # convert from meters to lat-lon:

    # myy = np.flipud(y)
    fig, axes = plt.subplots(3,2, figsize = (18, 24))
    # cm0 = axes[0,0].pcolormesh(x, invy, ele_field, shading = shading)

    # vmax = np.quantile(np.ravel(ele_field), 0.98)
    # vmin = np.quantile(np.ravel(ele_field), 0.02)
    cm0 = axes[0,0].pcolormesh(myx, myy, np.flipud(np.rot90(ele_field)), shading = shading)
    axes[0,0].set_xlabel(r'$x$ EAST [km]')
    axes[0,0].set_ylabel(r'$y$ NORTH [km]')
    axes[0,0].set_title('Terrain elevation')
    fig.colorbar(cm0, ax = axes[0,0], label = r'Elevation [$\mathrm{m}$ m.s.l.]')


    vmax = np.quantile(np.ravel(fFMdir_field), 0.98)
    # vmin = np.quantile(np.ravel(fFMdir_field), 0.02)
    cm1 = axes[0,1].pcolormesh(myx, myy, np.flipud(np.rot90(fFMdir_field)),
                               shading = shading, vmax=vmax)
    axes[0,1].set_xlabel(r'$x$ EAST [km]')
    axes[0,1].set_ylabel(r'$y$ NORTH [km]')
    axes[0,1].set_title('Direct flux normalized difference')
    # cm1.set_clim(0, 5)
    fig.colorbar(cm1, ax = axes[0,1], label = r'$f_{dir}$', extend='max')
    # fig.colorbar(cm1, ax = axes[0,1], extend = 'max')


    vmax = np.quantile(np.ravel(fFMdif_field), 0.98)
    # vmin = np.quantile(np.ravel(fFMdif_field), 0.02)
    cm2 = axes[1,0].pcolormesh(myx, myy, np.flipud(np.rot90(fFMdif_field)),
                               shading = shading, vmax=vmax)
    axes[1,0].set_xlabel(r'$x$ EAST [km]')
    axes[1,0].set_ylabel(r'$y$ NORTH [km]')
    axes[1,0].set_title(r'Diffuse flux normalized difference')
    # cm2.set_clim(0, 5)
    fig.colorbar(cm2, ax = axes[1,0], label = r'$f_{dif}$', extend='max')
    # fig.colorbar(cm2, ax = axes[1,0], extend = 'max')

    vmax = np.quantile(np.ravel(fFMrdir_field), 0.98)
    # vmin = np.quantile(np.ravel(fFMrdir_field), 0.02)
    cm3 = axes[1,1].pcolormesh(myx, myy, np.flipud(np.rot90(fFMrdir_field)),
                        shading = shading, vmax = vmax)
    axes[1,1].set_xlabel(r'$x$ EAST [km]')
    axes[1,1].set_ylabel(r'$y$ NORTH [km]')
    axes[1,1].set_title(r'Reflected-direct flux fraction')
    # cm3.set_clim(0, 2)
    fig.colorbar(cm3, ax = axes[1,1], label = r'$f_{rdir}$', extend='max')
    # fig.colorbar(cm3, ax = axes[1,1], extend = 'max')


    vmax = np.quantile(np.ravel(fFMrdif_field), 0.98)
    # vmin = np.quantile(np.ravel(fFMrdif_field), 0.02)
    cm4 = axes[2,0].pcolormesh(myx, myy, np.flipud(np.rot90(fFMrdif_field)),
                    shading = shading, vmax = vmax)
    axes[2,0].set_xlabel(r'$x$ EAST [km]')
    axes[2,0].set_ylabel(r'$y$ NORTH [km]')
    axes[2,0].set_title(r'Reflected-diffuse flux fraction')
    # cm4.set_clim(0, 2)
    fig.colorbar(cm4, ax = axes[2,0], label = r'$f_{rdif}$', extend='max')


    vmax = np.quantile(np.ravel(fFMcoup_field), 0.98)
    # vmin = np.quantile(np.ravel(fFMcoup_field), 0.02)
    cm5 = axes[2,1].pcolormesh(myx, myy, np.flipud(np.rot90(fFMcoup_field)),
                               shading=shading, vmax=vmax)
    axes[2,1].set_xlabel(r'$x$ EAST [km]')
    axes[2,1].set_ylabel(r'$y$ NORTH [km]')
    axes[2,1].set_title(r'Coupled flux normalized difference')
    # cm5.set_clim(0, 8)
    fig.colorbar(cm5, ax = axes[2,1], label = r'$f_{coup}$', extend='max')


    # outfigdir = os.path.join('/home/enrico/Documents/dem_datasets/', 'outfigdir')
    plt.tight_layout()
    # plt.savefig(os.path.join(outfigdir,
    #         'output_differences_ave_{}.png'.format(aveblock)), dpi = 300)
    plt.savefig(os.path.join(outfigdir,
                'label_fields_cosz_{:0.2f}_ave_{:d}_adir_{:0.2f}_phi_{:0.2f}.png'.format(
                cosz, trunc(aveblock), adir, phi)), dpi = 300)
    plt.close()



    fig, axes = plt.subplots(3,2, figsize = (18, 24))
    # cm0 = axes[0,0].pcolormesh(x, invy, ele_field, shading = shading)
    cm0 = axes[0,0].pcolormesh(myx, myy, np.flipud(np.rot90(ele_field)), shading = shading)
    axes[0,0].set_xlabel(r'$x$ EAST [km]')
    axes[0,0].set_ylabel(r'$y$ NORTH [km]')
    axes[0,0].set_title('Terrain elevation [m msl]')
    fig.colorbar(cm0, ax = axes[0,0])

    vmax = np.quantile(np.ravel(dim_fFMdir_field), 0.98)
    cm1 = axes[0,1].pcolormesh(myx, myy, np.flipud(np.rot90(dim_fFMdir_field)),
                               shading=shading, vmax=vmax)
    axes[0,1].set_xlabel(r'$x$ EAST [km]')
    axes[0,1].set_ylabel(r'$y$ NORTH [km]')
    axes[0,1].set_title('Direct flux normalized difference [-]')
    # cm1.set_clim(0, 5)
    fig.colorbar(cm1, ax = axes[0,1], label = r'$f_i$')
    # fig.colorbar(cm1, ax = axes[0,1], extend = 'max')

    vmax = np.quantile(np.ravel(dim_fFMdif_field), 0.98)
    cm2 = axes[1,0].pcolormesh(myx, myy, np.flipud(np.rot90(dim_fFMdif_field)),
                               shading=shading, vmax=vmax)
    axes[1,0].set_xlabel(r'$x$ EAST [km]')
    axes[1,0].set_ylabel(r'$y$ NORTH [km]')
    axes[1,0].set_title(r'Diffuse flux normalized difference [-]')
    # cm2.set_clim(0, 5)
    fig.colorbar(cm2, ax = axes[1,0], label = r'$f_i$', extend = 'max')
    # fig.colorbar(cm2, ax = axes[1,0], extend = 'max')

    vmax = np.quantile(np.ravel(dim_fFMrdir_field), 0.98)
    cm3 = axes[1,1].pcolormesh(myx, myy, np.flipud(np.rot90(dim_fFMrdir_field)),
                               shading=shading, vmax=vmax)
    axes[1,1].set_xlabel(r'$x$ EAST [km]')
    axes[1,1].set_ylabel(r'$y$ NORTH [km]')
    axes[1,1].set_title(r'Reflected-direct flux fraction [-]')
    # cm3.set_clim(0, 2)
    fig.colorbar(cm3, ax = axes[1,1], label = r'$f_i$', extend = 'max')
    # fig.colorbar(cm3, ax = axes[1,1], extend = 'max')


    vmax = np.quantile(np.ravel(dim_fFMrdif_field), 0.98)
    cm4 = axes[2,0].pcolormesh(myx, myy, np.flipud(np.rot90(dim_fFMrdif_field)),
                               shading=shading, vmax=vmax)
    axes[2,0].set_xlabel(r'$x$ EAST [km]')
    axes[2,0].set_ylabel(r'$y$ NORTH [km]')
    axes[2,0].set_title(r'Reflected-diffuse flux fraction [-]')
    # cm4.set_clim(0, 2)
    fig.colorbar(cm4, ax = axes[2,0], label = r'$f_i$', extend = 'max')

    vmax = np.quantile(np.ravel(dim_fFMcoup_field), 0.98)
    cm5 = axes[2,1].pcolormesh(myx, myy, np.flipud(np.rot90(dim_fFMcoup_field)),
                               shading=shading, vmax=vmax)
    axes[2,1].set_xlabel(r'$x$ EAST [km]')
    axes[2,1].set_ylabel(r'$y$ NORTH [km]')
    axes[2,1].set_title(r'Coupled flux normalized difference [-]')
    # cm5.set_clim(0, 8)
    fig.colorbar(cm5, ax = axes[2,1], label = r'$f_i$', extend = 'max')


    # outfigdir = os.path.join('/home/enrico/Documents/dem_datasets/', 'outfigdir')
    plt.tight_layout()
    # plt.savefig(os.path.join(outfigdir,
    #         'output_differences_ave_{}.png'.format(aveblock)), dpi = 300)
    plt.savefig(os.path.join(outfigdir,
                             'dimensional_fields_cosz_{:0.2f}_ave_{:d}_adir_{:0.2f}_phi_{:0.2f}.png'.format(
                                 cosz, trunc(aveblock), adir, phi)), dpi = 300)
    plt.close()



    fig, axes = plt.subplots(2,2, figsize = (18, 18))
    # cm0 = axes[0,0].pcolormesh(x, invy, ele_field, shading = shading)
    cm0 = axes[0,0].pcolormesh(myx, myy, np.flipud(np.rot90(ele_field)), shading = shading)
    axes[0,0].set_xlabel(r'$x$ EAST [km]')
    axes[0,0].set_ylabel(r'$y$ NORTH [km]')
    axes[0,0].set_title(r'Terrain elevation [$\mathrm{m}$ m.s.l.]')
    fig.colorbar(cm0, ax = axes[0,0])

    vmax = np.quantile(np.ravel(sian_field), 0.98)
    vmin = np.quantile(np.ravel(sian_field), 0.02)
    cm1 = axes[0,1].pcolormesh(myx, myy, np.flipud(np.rot90(sian_field)),
                    shading = shading, vmin=vmin, vmax=vmax)
    axes[0,1].set_xlabel(r'$x$ EAST [km]')
    axes[0,1].set_ylabel(r'$y$ NORTH [km]')
    axes[0,1].set_title(r'Solar incidence angle $\tilde{\mu}_i$')
    # cm1.set_clim(0, 5)
    fig.colorbar(cm1, ax = axes[0,1], extend = 'both')

    vmax = np.quantile(np.ravel(svfn_field), 0.98)
    vmin = np.quantile(np.ravel(svfn_field), 0.02)
    cm2 = axes[1,0].pcolormesh(myx, myy, np.flipud(np.rot90(svfn_field)),
                    shading = shading, vmin = vmin, vmax = vmax)
    axes[1,0].set_xlabel(r'$x$ EAST [km]')
    axes[1,0].set_ylabel(r'$y$ NORTH [km]')
    axes[1,0].set_title(r'Sky view $\tilde{V}_{d}$')
    # cm2.set_clim(0, 5)
    fig.colorbar(cm2, ax = axes[1,0], extend = 'both')

    vmax = np.quantile(np.ravel(tcfn_field), 0.98)
    vmin = np.quantile(np.ravel(tcfn_field), 0.02)
    cm3 = axes[1,1].pcolormesh(myx, myy, np.flipud(np.rot90(tcfn_field)),
                    shading = shading, vmin = vmin, vmax = vmax)
    axes[1,1].set_xlabel(r'$x$ EAST [km]')
    axes[1,1].set_ylabel(r'$y$ NORTH [km]')
    axes[1,1].set_title(r'Terrain configuration $\tilde{C}_{t}$')
    # cm3.set_clim(0, 2)
    fig.colorbar(cm3, ax = axes[1,1], extend = 'both')

    # outfigdir = os.path.join('/home/enrico/Documents/dem_datasets/', 'outfigdir')
    plt.tight_layout()
    # plt.savefig(os.path.join(outfigdir,
    #         'output_differences_ave_{}.png'.format(aveblock)), dpi = 300)
    plt.savefig(os.path.join(outfigdir,
                             'terrain_fields_cosz_{:0.2f}_ave_{:d}_adir_{:0.2f}_phi_{:0.2f}.png'.format(
                                 cosz, trunc(aveblock), adir, phi)), dpi = 300)
    plt.close()
    return



# def plot_gof_measures(file_res, outfigdir, METRICS_TO_PLOT = None, crossval_file=None):
def plot_gof_measures(file_res, outfigdir, METRICS_TO_PLOT = None, MODELS_TO_PLOT = None):
    # ds -> xarray dataset with the model's performance metrics
    # outfigdir -> rirectory where to store the plot
    # if crossval file is not None, plot also
    ds = xr.open_dataset(file_res)

    # if crossval_file is not None:
    #     ds_cv = xr.open_dataset(crossval_file)

    # fluxes = ds['FLUXES'].values
    COSZs = ds['COSZs'].values
    ncosz = len(COSZs)

    # LABELS = ds['LABELS'].values
    LABELS = ds.coords['label'].values
    AVEBLOCKS = ds.coords['aveblock'].values*90/1000 # scale in [km]
    DOMAINS = ds.coords['domain'].values
    # LABELS = ['fdir', 'fdif', 'frdirn', 'frdifn', 'fcoupn']
    LABEL_SYMB = [r'$f_{dir}$', r'$f_{dif}$', r'$f_{rdir}$', r'$f_{rdif}$', r'$f_{coup}$' ]
    nlabels = len(LABELS)
    MODELS0 = ds.coords['model'].values
    # matplotlib.use('Qt5Agg') # dyn show plots
    # plt.figure()

    # iadir = 0
    iadir = 0
    pad = 7
    figsize = 3.0

    # create a different plot for each metric
    # METRICS = ds['METRICS'].values

    # METRICS = ['R2']
    # METRIC_SYMB = [r'$R^2$']

    # list of all allowed metrics and their corresponding labels
    METRICS0 = ['R2', 'MAE', 'MSE','EVAR' ,'MAXE', 'MAPE']
    METRIC_SYMB0 = [r'$R^2$', 'MAE', 'MSE', 'EVAR', 'MAXE', 'MAPE']

    METRICS = [x for x in METRICS0 if x in METRICS_TO_PLOT]
    METRIC_SYMB = [ METRIC_SYMB0[xi] for xi in range(len(METRICS0)) if METRICS0[xi] in METRICS_TO_PLOT]
    MODELS = [ MODELS0[xi] for xi in range(len(MODELS0)) if MODELS0[xi] in MODELS_TO_PLOT]
    if len(METRICS) < 1:
        raise Exception('Valid metrics must be provided!')
    print("metrics to plot = {}".format(METRICS_TO_PLOT))
    print("models to plot = {}".format(MODELS_TO_PLOT))

    assert len(MODELS) < 5
    model_colors = ['blue', 'red', 'orange', 'green']
    ss_model_markers = ['-*', '-o', '-s', '-^']
    cv_model_markers = ['--*', '--o', '--s', '--^']

    training_domain = ds.attrs['cv_training_domain']


    for im, mymetric in enumerate(METRICS):
        for idom, mydomain in enumerate(DOMAINS):
            fig, axes = plt.subplots(nrows=ncosz, ncols=nlabels, figsize = (1.8*figsize*nlabels, figsize*ncosz))

            for ic, mycosz in enumerate(COSZs):
                for il, mylabel in enumerate(LABELS):

                    ds_plot = ds['GOFs'].isel(buffer=0, iadir=0, metric=im,
                                              rf_n_estimators = 0, rf_max_depth = 0)

                    # if crossval_file is not None:
                    #     ds_cv_plot = ds_cv['GOFs'].isel(buffer=0, iadir=0, metric=im,
                    #                               rf_n_estimators=0, rf_max_depth=0)

                    for imod, mymodel in enumerate(MODELS):

                        # same sample, continuous lines::
                        if training_domain == 'None':
                            # print(training_domain)
                            axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=imod,label=il)).values,
                                          ss_model_markers[imod], linewidth = 3.0, color = model_colors[imod], label = MODELS[imod])
                            # pass

                        else:
                            # print(training_domain)
                            # plot same sample data in continuous lines
                            axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(icosz=ic, label=il, model=imod)).sel(domain=training_domain).values,
                                              ss_model_markers[imod], linewidth = 3.0, color = model_colors[imod], label = '{}, SS'.format(MODELS[imod]))
                            # plot cross validation data in dashed lines
                            axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=imod,label=il)).values,
                                              cv_model_markers[imod], linewidth = 3.0, color = model_colors[imod], label = '{}, CV'.format(MODELS[imod]))

                    # # idom = 0
                    # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=0,label=il)).values, '-*', linewidth = 3.0, color = 'green', label = MODELS[0])
                    # # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=0,label=il)).values, '-*', linewidth = 3.0, color = 'red' )
                    # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=1,label=il)).values, '--', linewidth = 3.0, color = 'red', label = MODELS[1])
                    # # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=1,label=il)).values, '--', linewidth = 3.0, color = 'red'  )
                    # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=2,label=il)).values, '-o', linewidth = 3.0, color = 'blue', label = MODELS[2])
                    # # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=2,label=il)).values, '-^', linewidth = 3.0, color = 'red'  )
                    # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=3,label=il)).values, '-o', linewidth = 3.0, color = 'orange', label = MODELS[3])
                    # # axes[ic, il].plot(AVEBLOCKS, ds_plot.isel(dict(domain=idom, icosz=ic, model=3,label=il)).values, '-^', linewidth = 3.0, color = 'red'  )

                    if mymetric in ['R2', 'EVAR']:
                        axes[ic, il].plot(AVEBLOCKS, np.ones(AVEBLOCKS.shape), '--k')
                    else:
                        axes[ic, il].plot(AVEBLOCKS, np.zeros(AVEBLOCKS.shape), '--k')


                    axes[ic,il].grid(True)
                    # axes[ic, il].set_xlabel('scale [npixels]')
                    # axes[ic, il].set_ylabel(r'$R^2$')
                    if ic == len(COSZs)-1 and il == 0:
                        # axes[ic, il].legend()
                        # axes[ic, il].legend(loc='upper right', ncol=5, bbox_to_anchor=(4.0, 1.85))  # legend only on first plot
                        axes[ic, il].legend(loc='lower center', ncol=2)
                        # axes[ic, il].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=len(MODELS))
                        # axes[ic, il].legend(loc='lower right', bbox_to_anchor=(0.5, -0.4), ncol=2)

                    # fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
                    #                     bottom=0.12)  # create some space below the plots by increasing the bottom-value
                    # axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

                    axes[ic, il].set_ylim([0,1.1])

                    if il == 0:
                        axes[ic, il].annotate( r'$\mu_0$ = {}'.format(COSZs[ic]),
                                               xy=(0, 0.5), xytext=(
                                -axes[ic, il].yaxis.labelpad - pad, 0),
                                               xycoords=axes[ic, il].yaxis.label,
                                               textcoords='offset points',
                                               size='large', ha='right', va='center')
                    if ic == 0:
                        axes[ic, il].set_title('{}'.format(LABEL_SYMB[il]))
                    # if ic == ncosz - 1:
                    #     axes[ic, il].set_xlabel(r'{}'.format(LABELS[il]))

                    if ic == ncosz - 1:
                        axes[ic, il].set_xlabel(r'scale [km]')
                    else:
                        axes[ic, il].xaxis.set_ticklabels([])
                        # axes[ic, il].set_xlabel([])

                    if il == 0:
                        # axes[ic, il].set_ylabel(r'$R^2$')
                        axes[ic, il].set_ylabel(METRIC_SYMB[im])
                    else:
                        axes[ic, il].yaxis.set_ticklabels([])
                        # axes[ic, il].set_ylabel([])

                # plt.plot(modeldata['R2'][dict(idom=1,ibuf=0,iadir=0,icosz=2,imod=0,ilab=0)].values)
            plt.tight_layout()

            # if crossval_file is not None:
            #     savename = 'metric_ss+cv.png'
            # else:

            test_only = ds.attrs['test_only']
            if test_only:
                prefit_model_domain = ds.attrs['cv_training_domain']
                save_prefit_model_aveblock = ds.attrs['cv_training_aveblock']
                savename = 'metric_{}_tested_{}_CV_trained_{}_{}.png'.format(
                        mymetric, mydomain, prefit_model_domain, save_prefit_model_aveblock)
            else:
                savename = 'metric_{}_{}_samesample.png'.format(mymetric, mydomain)

            plt.savefig(os.path.join(outfigdir, savename), dpi = 300)
            plt.close()

    return


def plot_rf_hyper_params(file_res, outfigdir_gof):

    ds = xr.open_dataset(file_res, outfigdir_gof)
    # n_mtde = len(mtde)
    # n_nest = len(nest)

    rf_outfigdir =os.path.join(outfigdir_gof,'random_forest_hyperparameters')
    os.system("mkdir -p {}".format(rf_outfigdir))

    nest = ds.coords['rf_n_estimators'].values
    mtde = ds.coords['rf_max_depth'].values
    ds_plot = ds['GOFs'].isel(buffer=0, aveblock = 0, iadir=0,
                              metric=0, icosz = 0, domain=0).sel(model='RFR', label = 'fdif')
    # ds_nest = ds_plot.isel(rf_max_depth=0)
    # ds_mtde = ds_plot.isel(rf_n_estimators = 0)

    plt.figure(figsize=(9,9))
    for int, nt in enumerate(mtde):
        ds_nest = ds_plot.isel(rf_max_depth=int)
        plt.plot(nest, ds_nest, '-o', label='max depth = {}'.format(nt))
    plt.legend()
    plt.xlabel('Number of estimators')
    plt.ylabel(r'RF gof metric $R^2$')
    plt.savefig(os.path.join(rf_outfigdir, 'R2_vs_n_estimators.png'))
    plt.close()


    plt.figure(figsize=(9,9))
    for inee, nee in enumerate(nest):
        ds_mtde = ds_plot.isel(rf_max_depth=inee)
        plt.plot(mtde, ds_mtde, '-o', label='Number of est = {}'.format(nee))
    plt.legend()
    plt.xlabel('Tree max depth')
    plt.ylabel(r'RF gof metric $R^2$')
    plt.savefig(os.path.join(rf_outfigdir, 'R2_vs_n_estimators.png'))
    plt.close()

    return
