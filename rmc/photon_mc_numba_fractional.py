
import os
import re
import copy
import time
import random
import numba as nb
from numba import jit
from numba.experimental import jitclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import atmosphere_old as atm
import photon_mc_atmosphere as atm
from itertools import product
import pandas as pd
import matplotlib
from netCDF4 import Dataset

from numba import types
from numba.typed import Dict


# @jit(nopython=True)
# def test_photon():
#     myph = Photon()
#     print(myph.energy)
#     myph.energy += 1.0
#     print(myph.energy)
#     # print(myph['energy'])
#     return myph


# def random_num():
#     return random.random()

@jit(nopython=True)
def random_num():
    # random.seed(1)
    myrand = random.random()
    # myrand = 0.5
    # print(myrand)
    return myrand

# @jit(nopython=True)
# def random_num():
#     myrand = 0.9
#     return myrand



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



def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result


spec_results = [
               ('x', nb.float64[:]),
               ('y', nb.float64[:]),
               ('EDIR',  nb.float32[:,:]),
               ('EDIF',  nb.float32[:,:]),
               ('ERDIR', nb.float32[:,:]),
               ('ERDIF', nb.float32[:,:]),
               ('ECOUP', nb.float32[:,:]),
               ('eTOA', nb.float64),
               ('eABS', nb.float64),
               ('eSRF', nb.float64)
               ]


@nb.experimental.jitclass(spec_results)
class Result:
    def __init__(self, x, y):
        nx = len(x)
        ny = len(y)
        self.eTOA = 0.0
        self.eABS = 0.0
        self.eSRF = 0.0
        # self.x = np.zeros(nx, dtype=np.float64)
        # self.y = np.zeros(ny, dtype=np.float64)
        self.x = x
        self.y = y
        # use float 32 to limit size of output (rem up. limit 2*10^9)
        self.EDIR = np.zeros((nx, ny),  dtype=np.float32)
        self.EDIF = np.zeros((nx, ny),  dtype=np.float32)
        self.ERDIR = np.zeros((nx, ny), dtype=np.float32)
        self.ERDIF = np.zeros((nx, ny), dtype=np.float32)
        self.ECOUP = np.zeros((nx, ny), dtype=np.float32)


spec_photon = [('verbose', nb.boolean),
        ('cosz', nb.float64),
        ('phi', nb.float64),
        ('S0', nb.float64[:]),
        ('S1', nb.float64[:]),
        ('last_impact', nb.float64[:]),
        ('last_normal', nb.float64[:]),
        ('energy', nb.float64),
        ('is_alive', nb.boolean),
        ('is_impacted', nb.boolean),
        #
        ('direct_hit', nb.boolean),
        ('direct', nb.boolean),
        ('reflected', nb.boolean),
        ('multiple_refl', nb.boolean),
        # ('nreflections', nb.int32),
        # ('nscattering', nb.int32),
        ('entered_domain', nb.boolean),
        ('crossing_bounds', nb.boolean),
        ('frac_energy', nb.boolean),
        ('aerosol', nb.boolean),
        ('brdf', nb.boolean),
        ('below', nb.int32),
        ('Etoa', nb.float64),
        ('Eatm', nb.float64),
        ('Esrf', nb.float64)]


@nb.experimental.jitclass(spec_photon)
class Photon:
    def __init__(self):
        self.verbose = False
        self.cosz = 0.0
        self.phi = 0.0
        self.S0 = np.array([0.0, 0.0, 0.0])
        self.S1 = np.array([0.0, 0.0, 0.0])
        self.last_normal = np.array([0.0, 0.0, 0.0])
        self.last_impact = np.array([0.0, 0.0, 0.0])
        self.energy = 0.0
        self.is_alive = False
        self.is_impacted = False
        # self.nreflections = 0
        # self.nscattering = 0
        self.direct_hit = False # 1st
        self.direct = True # 2nd opposite
        self.reflected = False
        self.multiple_refl = False
        # self.first_dir_refl = False
        # self.impacted = False # BIT 1 in CHEN 2006
        # self.scattered = False # BIT 2 in CHEN 2006
        self.entered_domain = False
        self.crossing_bounds = False
        self.frac_energy = False
        self.aerosol = False
        self.brdf = False
        self.below = 0
        self.Etoa = 0.0
        self.Eatm = 0.0
        self.Esrf = 0.0

        # myresult.eSRF += photon.Esrf
        # myresult.eABS += photon.Eatm
        # myresult.eTOA += photon.Etoa

    # def __getitem__(self, key):
    #     print("Inside `__getitem__` method!")
    #     return getattr(self, key)







@jit(nopython=True)
def mydot(x, y, n=3):
    myd = 0.0
    for i in range(n):
        myd += x[i]*y[i]
    return myd


@jit(nopython=True)
def generate_photon(x, y, hTOA=66000, cosz0=-1, phi0=0,
            verbose=False, zelevmax = 0, tilted = True,
            frac_energy = True, aerosol = True, brdf=False):
    # start the photon in a random position within the (TOA) domain

    xTOP0, xTOPL, yTOP0, yTOPL = upperbounds(hTOA, zelevmax,
                               x, y, cosz0, phi0, tilted=tilted)
    # random.seed(10)
    x0 = xTOP0 + random_num() * (xTOPL - xTOP0)
    # x0 = xTOP0 + random.random() * (xTOPL - xTOP0)
    # x0 = xTOP0 + 0.5 * (xTOPL - xTOP0)
    # random.seed(10)
    y0 = yTOP0 + random_num() * (yTOPL - yTOP0)
    # y0 = yTOP0 + 0.5 * (yTOPL - yTOP0)
    z0 = hTOA

    # photon = {}
    # photon['verbose'] = verbose
    # photon['cosz'] = cosz0  # current zenith angle cosine
    # photon['phi'] = phi0  # current azimuth angle
    # photon['S0'] = np.array([x0, y0, z0])
    # photon['S1'] = np.array([x0, y0, z0])
    # photon['energy'] = 1.0  # start with a unitary energy packet
    # photon['is_alive'] = True  # is it still alive?
    # photon['is_impacted'] = False  # did it recently hit the surface?
    # photon['nreflections'] = 0  # number of times it was reflected by surfaces
    # photon['nscattering'] = 0  # number of times it was scattered in the air
    # photon['entered_domain'] = False  # only for tilted TOA domain
    # photon['direct'] = True  # is it still direct beam (True) or diffuse (False)
    # photon['crossing_bounds'] = False
    # photon['below'] = 0
    # photon['Etoa'] = 0
    # photon['Eatm'] = 0
    # photon['Esrf'] = 0

    photon = Photon()
    photon.verbose = verbose
    photon.cosz = cosz0  # current zenith angle cosine
    photon.phi = phi0  # current azimuth angle
    photon.S0 = np.array([x0, y0, z0])
    photon.S1 = np.array([x0, y0, z0])
    photon.energy = 1.0  # start with a unitary energy packet
    photon.is_alive = True  # is it still alive?
    photon.is_impacted = False  # did it recently hit the surface?
    # photon.nreflections = 0  # number of times it was reflected by surfaces
    # photon.nscattering = 0  # number of times it was scattered in the air
    photon.direct_hit = False  # 1st
    photon.direct = True  # 2nd opposite
    photon.reflected = False
    photon.multiple_refl = False
    photon.entered_domain = False  # only for tilted TOA domain
    # photon.direct = True  # is it still direct beam (True) or diffuse (False)
    photon.crossing_bounds = False
    photon.frac_energy = frac_energy
    photon.aerosol = aerosol
    photon.below = 0
    photon.Etoa = 0.0
    photon.Eatm = 0.0
    photon.Esrf = 0.0
    photon.brdf = brdf

    if photon.verbose:
        print('--------------------------------------------------------')
        print('generate photon in S0 = ', photon.S0)
        print('TOA domain boundaries:')
        print('x: from - to ', xTOP0, xTOPL)
        print('y: from - to ', yTOP0, yTOPL)
        print('--------------------------------------------------------')
    return photon


@jit(nopython=True)
def travel(photon, dfatm=None):

    if photon.verbose:
        print('photon: travelling from S0 to S1!')
        print('before travel: S0 = ', photon.S0)
        print('before travel: S1 = ', photon.S1)
    # photon['is_impacted'] = False # remove the impact tracer when moving
    # photon['S0'] = photon['S1'].copy()
    photon.is_impacted = False # remove the impact tracer when moving
    photon.S0 = photon.S1.copy()
    # l = compute_travel_distance(
    #     z0=photon['S0'][2], cosz=photon['cosz'],
    #     dfatm=dfatm, verbose=photon['verbose'])[0]

    # l = compute_travel_distance(
    #     z0=photon.S0[2], cosz=photon.cosz,
    #     dfatm=dfatm, verbose=photon.verbose)[0]

    restd = compute_travel_distance(photon, dfatm=dfatm)
    l = restd[0] # distance travelled
    # photon = restd[-1] # photon object with updated energy
    # sintheta = np.sqrt(1-photon['cosz']**2)
    # photon['S1'][0] = photon['S0'][0] + l * sintheta*np.sin(photon['phi'])
    # photon['S1'][1] = photon['S0'][1] + l * sintheta*np.cos(photon['phi'])
    # photon['S1'][2] = photon['S0'][2] + l * photon['cosz']

    sintheta = np.sqrt(1.0 - photon.cosz**2)
    photon.S1[0] = photon.S0[0] + l * sintheta*np.sin(photon.phi)
    photon.S1[1] = photon.S0[1] + l * sintheta*np.cos(photon.phi)
    photon.S1[2] = photon.S0[2] + l * photon.cosz
    if photon.verbose:
        print('photon: travelling from S0 to S1!')
        print('after travel: S0 = ', photon.S0)
        print('after travel: S1 = ', photon.S1)
        print('-----------------------------------------------------------')
    return photon


@jit(nopython=True)
def check_impacts(photon, x, y, Z, adaptive=False,
                  nb1=10, nb2=6, nb3=6, dfatm=None):
    # check whether there are impacts in the segment S0 - S1:
    # first check the bounding box
    # then the horizontal projection
    # then the 3D intersections
    # get that with minimum distance from S1
    if photon.verbose:
        # init_time = time.time()
        print('checking photon impacts on the surface in:')
        print(photon.S0)
        print(photon.S1)
        # print('-------------------------------------------')

    if adaptive:
        # be careful to chose nb1, nb2 nb2. Choose them beforehand
        # for a given grid size and test that is works. It does not work if
        # nb1, nb2, nb3 are too large for too small a domain.

        # This option is generally slower so let's use the 3-fold code
        # IS_INT, MYNORM, INTPOINT = adaptive_intersection_2(
        #       photon.S0, photon.S1, x, y, Z, nb1=16, nb2=4)

        # IS_INT_0, MYNORM_0, INTPOINT_0 = check_surface_intersections(
        #     photon.S0, photon.S1,
        #     x, y, Z, usebox=False, check_2d=False)

        IS_INT, MYNORM, INTPOINT = adaptive_intersection_3(
              photon.S0, photon.S1, x, y, Z, nb1=nb1, nb2=nb2, nb3=nb3)
                # photon.S0, photon.S1, x, y, Z, nb1 = 1, nb2 = 1, nb3 = 1)

        # if IS_INT and IS_INT_0:
        #     print('*********************')
        #     mydiff = np.abs(INTPOINT_0[0] - INTPOINT[0])
        #     print('mydiff', mydiff)
        #     print('*********************')
        #     if mydiff > 1e-6:
        #         print('Error: large impact difference!')
        #     assert IS_INT == IS_INT_0
        #     assert mydiff < 1e-6

    else:
        # non adaptive algorithm, can be awfully slow in finding intersections.
        # res = check_surface_intersections(photon['S0'], photon['S1'],
        #                 x, y, X, Y, Z, usebox=True, check_2d=False)

        # DO NOT USE 'usebox' or 'check_2d' options - deprecated!
        IS_INT, MYNORM, INTPOINT = check_surface_intersections(
                photon.S0, photon.S1,
                x, y, Z, usebox=False, check_2d=False)

    # print(IS_INT)
    # print(MYNORM)
    # print(INTPOINT)

    if IS_INT: # intersection with a surface found
        eps = 1E-6
        photon.is_impacted = True
        # track the case of first reflection of direct beam::
        if not photon.reflected and photon.direct:
            photon.direct_hit = True
        photon.last_impact = INTPOINT
        photon.last_normal = MYNORM
        photon.S1[0] = INTPOINT[0]
        photon.S1[1] = INTPOINT[1]
        photon.S1[2] = INTPOINT[2] + eps # make sure it stays above

        # print('removing absorbed energy')
        if photon.frac_energy:
            path_absorption(photon, dfatm=dfatm)

        if photon.verbose:
            print('found surface intersection in P = ', photon.S1)
    else: # no surface intersection found
        photon.is_impacted = False
        # photon['is_impacted'] = False
        if photon.verbose:
            print('photon: no surface intersection found between s0 and S1:')
            print('S0 = ', photon.S0)
            print('S1 = ', photon.S1)
            print('-------------------------------------------------------')
    # if photon.verbose:
    #     print('check impacts: time elapsed = {}'.format((time.time()-init_time)/60.0))
    #     print('-------------------------------------------------------')
    return photon


@jit(nopython=True)
def check_reflection(photon, myresult,
                     alpha_dir=None, alpha_dif=None,
                     adirmat=None, adifmat=None, xalb=None, yalb=None,
                     const_albedo = True):
    # if impact: check if absorption or reflection
    # compute the fraction of energy lost (alpha*energy)
    # alpha = terrain albedo
    # self.energy = alpha * self.energy
    # transferred = self.energy * (1 - alpha)
    # store this somewhere in the 2D MAP
    # instead, transfer or maintain entire energy:
    # TYPE: can be 'LAMB' or 'BRDF (TO DO)


    if not const_albedo:
        # look up the albedo values closest to the point of interest ::
        indx = np.argmin( np.abs(xalb - photon.S1[0]))
        indy = np.argmin( np.abs(yalb - photon.S1[1]))
        alpha_dir = adirmat[indx, indy]
        alpha_dif = adifmat[indx, indy]
    # else, they remain equal to those provided

    # if photon.brdf:
        # Use the BRDF formulation as in Schaaf at al., 2002
        # for direct (=black sky) and diffuse (=white sky) light
        # must input 3 geographically variable parameters


    # if photon.brdf: # FOR SNOW:
    #     pass
    #     # see Larue et al., 2020,
    #     SSA = 4.5 # [m^2 kg^-1] # snow specific surface area (~oldish snow here)
    #     rho_snow = 917.0 #[kg m^-3] snow density at 0^C
    #     B = 1.6
    #     g = 0.86
    #     frac_dif = np.sqrt(2 * B * gamma_lambda)/ (3*rho_snow*SSA*(1-g))
    #     alpha_dif = np.exp(-4*frac_dif)
    #
    #     term1 = -12/7*(1 + 2*np.costhetas)
    #     # theta2 = np.sqrt( (2*B*gamma_lambda)/())
    #     alpha_dir = np.exp(term1 * term2)

    if not photon.direct:
        alpha = alpha_dif
    else:
        alpha = alpha_dir

    # add case of variable alpha in the domain
    # if not already reflecte"
    # if not photon.reflected:

    UZHAT = np.array([0.0, 0.0, 1.0]) # upward direction
    # UNORM = photon['last_normal'] # normal to the surface (upward oriented)

    UNORM = np.array([0.0, 0.0, 0.0]) # most recent impact's surface normal
    for ix3 in range(3):
        UNORM[ix3] = photon.last_normal[ix3]
    # UNORM = photon.last_normal # normal to the surface (upward oriented)

    if np.dot(UNORM, UZHAT) < 0.0:
        raise Exception('Error in check_reflection: Surface normal downward!')

    xi = random_num()

    if xi > alpha and not photon.frac_energy: # photon absorbed at the surface!
        photon.is_alive = False
        photon.Esrf = 1.0
        update_energy(photon, myresult, 1.0)
        if photon.verbose:
            print('check_reflection: surface absorption '
                  'in P = ', photon.S1)
            print('-------------------------------------------------------')

    else: # photon reflected by the surface, or fractional energy case
        xi1 = random_num()
        xi2 = random_num()
        coszr = np.sqrt(xi1)
        phir = 2.0 * np.pi * xi2

        # sinzr = np.sqrt(1.0-coszr**2)
        # cosphir = np.cos(phir)
        # sinphir = np.sin(phir)
        # vn, t1n, t2n =twonormal(UNORM)
        # # compute the components of the new direction vector:
        # v1 = coszr*vn
        # v2 = sinzr*cosphir*t1n
        # v3 = sinzr*sinphir*t2n
        # vnew = v1 + v2 + v3
        # # compute polar coords from this:
        # r1, cosz1, phi1 = cart2polar(x=vnew[0], y=vnew[1], z=vnew[2])


        cosz1, phi1 = rotate_direction(coszr=coszr, phir=phir, VEC=UNORM)
        photon.cosz = cosz1 # azimuth = - diff angle
        photon.phi = phi1

        if photon.frac_energy:
            if photon.verbose:
                print('energy before reflection ', photon.energy)
            entran = photon.energy * (1 - alpha)
            photon.Esrf = photon.Esrf + entran
            update_energy(photon, myresult, entran)

            photon.energy = photon.energy * alpha

        # photon.nreflections += 1
        # update the reflection state for the photon::
        if not photon.reflected:
            photon.reflected = True
        else:
            photon.multiple_refl = True



        if photon.verbose:
            print('check_reflection: surf. reflection in P = ', photon.S1)
            print('reflected from surface with normal = ', UNORM)
            # print('dot product with new direction = ', np.dot(UNORM, vnew))
            print('changing cosz to ', photon.cosz)
            print('changing phi to ', photon.phi)
            print('energy after ', photon.energy)
            print('-------------------------------------------------------')
    return photon


@jit(nopython=True)
def check_scattering(photon, forcepscatter = False, pscatterf=0, dfatm=None):
    # if no impacts in the segment S0-S1
    # determine if the event is scattering or absorption
    # pscat = 0.9  # add here function of photon position (x, y, z)

    # PROBABILITY OF SCATTERING DEP ON SINGLE SCATTERING ALBEDO I.E.
    # RATIO OF SCATTERING CROSS SEC. AND EXTINCTION CROSS SECTION
    # pscatter = SSAz # probability of scattering = single scattering albedo
    # pabs = 1 - pscatter # probability of absorption

    # frac_energy = True

    if forcepscatter:
        print("check_scattering WARNING: using the fictitious pscatterf \n "
              "single scattering albedo - FOR TESTING PURPOSES ONLY!!!")
        pscatter=pscatterf
        prayleigh = 1.0
    else:

        # MODIFIED - FRACEN -> always scattering
        # pscatter = get_atm_value(photon['S0'][2],
        #                          param = 'wc', dfatm = dfatm)
        # modified from S0 to S1 here

        # compute scattering probability in the case of NO fractional energy
        # if photon.aerosol:
        #     ext_coeff = 'k_ext_tot' # total coefficient (gas + aerosol)
        #     sca_coeff = 'k_sca_tot' # total coefficient (gas + aerosol)
        # else:
        #     ext_coeff = 'k_ext_gas' # gas only scattering coefficient
        #     sca_coeff = 'k_sca_gas' # gas only scattering coefficient

        # kscatter_ray = get_atm_value(photon.S1[2],
        #                              param = 'extb_gas', dfatm = dfatm)
        #
        # kscatter_aer = get_atm_value(photon.S1[2],
        #                              param = 'extb_aer', dfatm = dfatm)



        # pscatter = get_atm_value(photon.S1[2],param = 'wc', dfatm = dfatm)
        #
        #
        # k = get_atm_value(photon.S1[2],
        #                          param = 'wc', dfatm = dfatm)

        # IF NO FRAC ENERGY:
        k_sca_gas = get_atm_value(photon.S1[2],param = 'k_sca_gas', dfatm = dfatm)
        k_sca_tot = get_atm_value(photon.S1[2],param = 'k_sca_tot', dfatm = dfatm)
        k_ext_tot = get_atm_value(photon.S1[2],param = 'k_ext_tot', dfatm = dfatm)
        pscatter = k_sca_tot / k_ext_tot

        if photon.aerosol:
            # probability of rayleigh conditional to scattering occurring
            prayleigh = k_sca_gas / (k_sca_tot + 1E-9) # IN BOTH CASES
        else:
            prayleigh = 1.0
            # in this case pscatter => k_sca_gas / k_ext_gas, but they are the same


    if photon.verbose:
        print('check_scattering: scattering or absorption?')
        print('before:')
        print('S0 = ', photon.S0)
        print('S1 = ', photon.S1)
        print('energy before = ', photon.energy)
        # print('prob scatter in S1 is = ', pscatter)
        print('if TRUE, fractional energy is active '
              '- > always scattering here', photon.frac_energy)
    # xis = random.random()

    xis = random_num() # probability of scattering event
    xits = random_num() # probability of different scattering types

    # prob_rayleigh_scat =
    # aerosol_scat = 1
    # aerosol_scat = 1

    if xis < pscatter or photon.frac_energy:
        # photon['direct'] = False  # scattering event
        photon.direct = False  # scattering event
        if photon.reflected:           # if scattering happens after a surface reflection
            photon.multiple_refl = True # we consider this (refl + scat) as coupled flux
        cosz0 = photon.cosz
        phi0 =  photon.phi
        # cosz0 = photon['cosz']
        # phi0 =  photon['phi']
        # sinphi0 = np.sin(phi0)
        # cosphi0 = np.cos(phi0)
        # generate angle deviation (in local coordinates)
        # xi1 = random.random() # RNG
        # xi2 = random.random() # RNG


        xi1 = random_num() # RNG
        xi2 = random_num() # RNG
        phir = 2.0 * np.pi * xi2

        # Ar = 8.0*(xi1 - 0.5) # ENRICO
        # ur = (0.5*(Ar + np.sqrt(Ar**2 + 4.0)))**(1.0/3.0)

        if xits <= prayleigh: # RAYLEIGH SCATTERING PHASE FUNCTION
            Ar = 4.0*xi1 - 2 # MEYER's q
            ur = (-Ar + np.sqrt(Ar**2 + 1.0))**(1.0/3.0)
            # ur = (0.5*(Ar - np.sqrt(Ar**2 + 4.0)))**(1.0/3.0)
            coszr = ur - 1.0/ur

        else: # HENYEY-GREENSTEIN PHASE FUNCTION
            # # add a switch here if adding clouds to the model
            g = get_atm_value(photon.S1[2], param='g_aer', dfatm=dfatm)
            term1 = (g**2 - 1)/(2*g*xi1 - g - 1)
            coszr = 1/2/g*(1 + g**2 - term1**2)


        # NOW ROTATE THE NEW ANGLES AROUND OLD DIRECTION VECTOR


        # sinphir = np.sin(phir)
        # cosphir = np.cos(phir)
        # sinzr = np.sqrt(1.0-coszr**2)
        VEC = polar2cart(r=1.0, cosz=cosz0, phi=phi0) # original direction unit vector
        # vn, t1n, t2n = twonormal(VEC)
        # # compute the components of the new direction vector:
        # v1 = coszr*vn
        # v2 = sinzr*cosphir*t1n
        # v3 = sinzr*sinphir*t2n
        # vnew = v1 + v2 + v3
        # r1, cosz1, phi1 = cart2polar(x=vnew[0], y=vnew[1], z=vnew[2])

        cosz1, phi1 = rotate_direction(coszr=coszr, phir=phir, VEC=VEC)
        photon.cosz = cosz1
        photon.phi = phi1
        # photon.nscattering += 1
        if photon.frac_energy:
            path_absorption(photon, dfatm=dfatm)

        if photon.verbose:
            print('photon: scattering in P = ', photon.S1)
            print('changing cosz to ', photon.cosz)
            print('changing phi to ' , photon.phi)
            print('-------------------------------------------------------')

    else: # no fractional energy & absorption is happaning
        photon.is_alive = False  # atmospheric absorption
        photon.Eatm = 1 # case of non-fractional energy - all energy is absorbed
        # photon['is_alive'] = False  # absorption
        # photon['Eatm'] = 1
        if photon.verbose:
            print('photon: absorption by atmosphere in P = ', photon.S1)
            print('-------------------------------------------------------')
    if photon.verbose:
        print('after check_scattering:')
        print('after:')
        print('S0 = ', photon.S0)
        print('S1 = ', photon.S1)
        print('energy after = ', photon.energy)
    return photon


@jit(nopython=True)
def rotate_direction(coszr=None, phir=None, VEC=None):
    """rotate a 3D array around another one (VEC)
    which can be the original photon direction (int he case of scattering)
    or the normal to the surface (in the case of reflection)
       of angles coszr, phir
    """
    sinphir = np.sin(phir)
    cosphir = np.cos(phir)
    sinzr = np.sqrt(1.0 - coszr ** 2)
    vn, t1n, t2n = twonormal(VEC)
    # compute the components of the new direction vector in the local frame:
    v1 = coszr * vn
    v2 = sinzr * cosphir * t1n
    v3 = sinzr * sinphir * t2n
    # v2 = sinzr * sinphir * t1n
    # v3 = sinzr * cosphir * t2n
    vnew = v1 + v2 + v3
    # r1, cosz1, phi1 = cart2polar(x=vnew[0], y=vnew[1], z=vnew[2])
    r1, new_cosz, new_phi = cart2polar(x=vnew[0], y=vnew[1], z=vnew[2])
    return new_cosz, new_phi


#

@jit(nopython=True)
def check_all_boundaries(photon, x, y, hTOA = 0, hMIN = 0,
                         tilted = False, cosz0=0, phi0=0, dfatm=None):
    # if tilted hMIN must be set equal to zelevmax!!!!!!
    #  IF TILTED, USE TILTED BEAM FROM TOA TO hMOUNT
    # cosz0, phi0 must be initial value of the simulation (sun) no current photon values!
    #hMIN must be set equal to zelevmax!!!!!

    eps = 1E-6 # safety margin to check boundary crossings
    if photon.verbose:
        print('check_all_boundaries: running for tilted domain = ', tilted)
    xTOA0, xTOAL, yTOA0, yTOAL = upperbounds(hTOA, hMIN,  # tilted stops here
                                x, y, cosz0, phi0, tilted = tilted)

    P0 =  np.array( [ x[0],  y[0],   hMIN ] )
    Py =  np.array( [ x[0],  y[-1],  hMIN ] )
    Px =  np.array( [ x[-1], y[0],   hMIN ] )
    Pxy = np.array( [ x[-1], y[-1],  hMIN ] )
    P0t = np.array( [ xTOA0, yTOA0,  hTOA ] )
    Pyt = np.array( [ xTOA0, yTOAL,  hTOA ] )
    Pxt = np.array( [ xTOAL, yTOA0,  hTOA ] )

    # CHECK INTERSECTION ALONG ALL 6 PLANES:
    PLANES = ['W', 'E', 'S', 'N', 'U', 'D'] # up and down too!

    # TRIPLET OF POINTS (=pair of vectors) WHICH INDIVIDUATE EACH PLANE:
    # Pi = {'W': [P0, Py, P0t],
    #       'E': [Px, Pxy, Pxt],
    #       'S': [P0, Px, P0t],
    #       'N': [Py, Pxy, Pyt],
    #       'U': [P0t, Pyt, Pxt],
    #       'D': [P0, Py, Px]
    #       }


    Pi0 = {'W': P0,
          'E':  Px,
          'S':  P0,
          'N':  Py,
          'U':  P0t,
          'D':  P0
          }

    Pi1 = {'W': Py,
           'E': Pxy,
           'S': Px,
           'N': Pxy,
           'U': Pyt,
           'D': Py
           }

    Pi2 = {'W': P0t,
          'E': Pxt,
          'S': P0t,
          'N': Pyt,
          'U': Pxt,
          'D': Px
          }

    ################
    #### PRELIMINARY: CHECK THAT s0 within domain
    # xz0s0, xzLs0, yz0s0, yzLs0 = upperbounds(photon['S0'][2],
    #                             hMIN, x, y, cosz0, phi0, tilted=tilted)
    # if (photon['S0'][0] < xz0s0 or photon['S0'][0] > xzLs0 or
    #            photon['S0'][1] < yz0s0 or photon['S0'][1] > yzLs0):
    #     raise Exception("Point S0 out of the domain!!")



    xz0s0, xzLs0, yz0s0, yzLs0 = upperbounds(photon.S0[2],
                     hMIN, x, y, cosz0, phi0, tilted=tilted)
    if (photon.S0[0] < xz0s0 or photon.S0[0] > xzLs0 or
            photon.S0[1] < yz0s0 or photon.S0[1] > yzLs0):
        raise Exception("Point S0 out of the domain!!")

    if photon.verbose:
        print('domain boudaries at hMIN:', xz0s0, xzLs0, yz0s0, yzLs0)

    INTERS = []  # which boundary plane is intersected
    INTERP = []  # intersection point
    # niter += 1
    # if niter >= nitermax:
    #     print('ERROR BOUNDARY:: MAXITER REACHED!')
    for pl in PLANES:
        # check intersection with the infinite plane
        # is_int, intP = intersect_plane(photon['S0'], photon['S1'],
        #                                Pi[pl][0], Pi[pl][1], Pi[pl][2])
        # is_int, intP = intersect_plane(photon.S0, photon.S1,
        #                                Pi[pl][0], Pi[pl][1], Pi[pl][2])

        is_int, intP = intersect_plane(photon.S0, photon.S1,
                                       Pi0[pl], Pi1[pl], Pi2[pl])
        # check whether such intersection points, if it exists, belongs to domain:
        if is_int:
            zlevel = intP[2]
            xz0, xzL, yz0, yzL = upperbounds(zlevel, hMIN, x, y, cosz0, phi0, tilted=tilted)
            condx = intP[1] > yz0 and intP[1] < yzL and intP[2] > hMIN and intP[2] < hTOA
            condy = intP[0] > xz0 and intP[0] < xzL and intP[2] > hMIN and intP[2] < hTOA
            condz = intP[0] > xz0 and intP[0] < xzL and intP[1] > yz0 and intP[1] < yzL

            if pl in ['E', 'W']:
                is_int_2 = condx
            elif pl in ['S', 'N']:
                is_int_2 = condy
            elif pl in ['U', 'D']:
                is_int_2 = condz
            else:
                raise Exception("Leaving from unknown boundary!")

        if is_int and is_int_2:
            INTERS.append(pl)
            INTERP.append(intP)

    if len(INTERS) == 0: # no boundary crossings found
        photon.crossing_bounds = False
        if photon.verbose:
            print('domain boudaries at current height level:', xz0,xzL,yz0,yzL)
            print('no boundary crossings found')

    elif len(INTERS) != 1:
        print('multiple crossings:', INTERS)
        # more than one crossing found - this should never happen!
        raise Exception("checking_all_boundaries Error: "
                        "More than one boundary crossing found!")

    else:
        # exactly one intersection found. Depending on which bound was crossed:
        if INTERS[0] == 'U': # PHOTON LEAVES THE ATMOSPHERE FROM ABOVE
            photon.crossing_bounds = False
            photon.is_alive = False

            photon.S1[0] = INTERP[0][0]
            photon.S1[1] = INTERP[0][1]
            photon.S1[2] = INTERP[0][2]
            if photon.frac_energy:
                path_absorption(photon, dfatm=dfatm) # EZDEV added
                photon.Etoa = photon.energy
            else:
                photon.Etoa = 1.0

            if photon.verbose:
                print('photon: leaving from the top of atmosphere')
                print('-----------------------------------------------')

        elif INTERS[0] == 'D' and not tilted:
            # this should never happen if not tilted domain
            # because surface impacts should have already been checked!
            photon.crossing_bounds = False
            photon.is_alive = False
            photon.below = 1
            raise Exception('Checking_all_boundaries Error:'
                            'photon leaving domain from lower boundary!')

            # else it can happen, because hMIN = zelevmax, no crossing yet
            # save the last position and leave
            # this case is treated below with the boundaries
        else:
            # self.crossing_bounds = True # start with true at the beginning
            # PHOTON IS LEAVING THE DOMAIN FROM A LATERAL BOUNDARY
            # OR from below in the tilted case only (hMIN = zelevmax)
            # COMPUTE THE NEW POSITION WHERE IT WILL REAPPEAR:
            # PNEW = get_periodic_bc(INTERP[0], x, y, Z, tbound=INTERS[0])
            # PNEW = Point(INTERP[0].x, INTERP[0].y, INTERP[0].z)
            PNEW = INTERP[0].copy() # same z-level
            xz0, xzL, yz0, yzL = upperbounds(INTERP[0][2],
                                hMIN, x, y, cosz0, phi0, tilted=tilted)
            if INTERS[0] == 'W':
                # PNEW.x = x[-1]  # other boundary
                PNEW[0] = xzL - eps
            if INTERS[0] == 'E':
                # PNEW.x = x[0]  # other boundary
                PNEW[0] = xz0 + eps
            if INTERS[0] == 'N':
                # PNEW.y = y[0]  # other boundary
                PNEW[1] = yz0 + eps  # other boundary
            if INTERS[0] == 'S':
                # PNEW.y = y[-1] # other boundary
                PNEW[1] = yzL - eps  # other boundary
            if INTERS[0] == 'D' and tilted:
                if photon.verbose:
                    print('Photon entering domain through zelevmax level')
                photon.entered_domain = True #
                photon.crossing_bounds = False # stop it in this case
                # if not absorbed by atmosphere, must pass through here
                # nothing to be done in the "from below" "tilted" case
                # in this case PNEW stays equal to the crossing point

            if photon.verbose:
                print('old start = ', photon.S0)
                print('old end = ', photon.S1)
                print('intersection = ', INTERP[0])
                print('new starting point = ', PNEW)

            # pay for photon path from old S0 to intersection point
            old_endpoint = photon.S1.copy()
            photon.S1 = INTERP[0].copy()
            if photon.frac_energy:
                path_absorption(photon, dfatm=dfatm)

            # then update the old point and the new
            photon.S0 = PNEW.copy()
            # TOTRAVEL = (photon.S1 - INTERP[0]).copy() # vector still to travel
            TOTRAVEL = old_endpoint - INTERP[0].copy() # vector still to travel
            photon.S1 = (PNEW + TOTRAVEL).copy()

            if photon.verbose:
                # print('checking boundaries iter = {}'.format(niter))
                print('checking boundaries')
                print('crossing boundary in', INTERS[0])
                print('new start S0 = ', photon.S0)
                print('new end S1 =', photon.S1)
            # check absence of surface intersections between S0 -> S1
            # if not (self.S0.z > zelevmax and self.S1.z > zelevmax):
            # self.check_impacts(x, y, X, Y, Z)
            # if self.is_impacFalse
            #     self.check_reflection(alpha)
            #     self.crossing_bounds = False
    if photon.verbose:
        print('-----------------------------------------------------------')
    return photon


@jit(nopython=True)
def launch(x=None, y=None, Z=None, hTOA=None, hMIN=None, cosz0=None, phi0=None,
           alpha_dir=None, alpha_dif=None,
           adirmat=None, adifmat=None, yalb=None, xalb=None,
           zelevmax = None, const_albedo=True, aerosol = True, brdf=False,
           frac_energy=True, myresult=None,
           adaptive=True, tilted=True,
           verbose = True, pscatterf=0, forcepscatter=False,
           nb1=10, nb2=6, nb3=6, dfatm=None):


    photon = generate_photon(x, y, hTOA=hTOA, cosz0=cosz0, phi0=phi0,
                    verbose=verbose, zelevmax=zelevmax, tilted=tilted,
                             frac_energy = frac_energy,
                             aerosol = aerosol, brdf=brdf)

    if tilted:
        niter0 = 0
        while photon.is_alive and not photon.entered_domain:
            niter0 += 1
            travel(photon, dfatm=dfatm)
            photon.crossing_bounds = True # set it here
            while photon.crossing_bounds and not photon.entered_domain:
                # must set hMIN = zelevmax when using tilted = True
                # no risk of surface impacts in the outer 'beam' domain
                # print('S0 = ', self.S0)
                # print('S1 = ', self.S1)
                check_all_boundaries(photon, x, y, hTOA=hTOA,
                                     hMIN=zelevmax, tilted=True,
                                     cosz0=cosz0, phi0=phi0, dfatm=dfatm)

                # if frac energy add absorption here?

                # check_all_boundaries(photon, x, y, Z, hTOA=hTOA, hMIN=zelevmax,
                #                      tilted=True, cosz0=cosz0, phi0=phi0,
                #                      enter_vertical=enter_vertical)

            # if photon['is_alive'] and not photon['entered_domain']:
            if photon.is_alive and not photon.entered_domain:
                # print('not entered domain yet - check scat')
                check_scattering(photon, forcepscatter=forcepscatter,
                                 pscatterf=pscatterf, dfatm=dfatm)
                # if scattered must travel next

            # travel(photon)
        # print('while niter0 = {}'.format(niter0))


    else: # not-tiled domain
        travel(photon, dfatm=dfatm)


    niter = 0
    while photon.is_alive:
        niter += 1
        check_impacts(photon, x, y, Z,
                      adaptive=adaptive,
                      nb1=nb1, nb2=nb2, nb3=nb3, dfatm=dfatm)
        if photon.is_impacted:

            # if frac energy add absorption here?
            check_reflection(photon, myresult,
                     alpha_dir=alpha_dir, alpha_dif=alpha_dif,
                     adirmat=adirmat, adifmat=adifmat, xalb=xalb, yalb=yalb,
                     const_albedo=const_albedo)
            # needs to travel next if alive
        else:
            photon.crossing_bounds = True  # set it here
            niter2 = 0
            while photon.is_alive and photon.crossing_bounds:  # until it keeps crossing laterally:
                niter2 += 1
                check_all_boundaries(photon, x, y, hTOA=hTOA,hMIN=hMIN,
                                      tilted = False, dfatm=dfatm)


                # check_all_boundaries_vert(photon, x, y, Z, hTOA=hTOA, hMIN=hMIN,
                #                      tilted=False) # ALWAYS FALSE HERE AFTER ENTRANCE!

                # do this only if at least one of S0, S1 is below zelevmax
                # if not (photon['S0'][2] > zelevmax and photon['S1'][2] > zelevmax):
                if not (photon.S0[2] > zelevmax and photon.S1[2] > zelevmax):
                    check_impacts(photon, x, y, Z,
                                adaptive=adaptive,
                                nb1=nb1, nb2=nb2, nb3=nb3, dfatm=dfatm)
                    # if photon['is_impacted']:
                    if photon.is_impacted:
                        # photon['crossing_bounds'] = False  # set it here
                        photon.crossing_bounds = False  # set it here
                        # if frac energy add absorption here?
                        check_reflection(photon, myresult,
                                         alpha_dir=alpha_dir,
                                         alpha_dif=alpha_dif,
                                         adirmat=adirmat, adifmat=adifmat,
                                         xalb=xalb, yalb=yalb,
                                         const_albedo=const_albedo)
                        # alpha_dir, alpha_dif,
                        #         adirmat, adifmat, x, y,
                        #         const_albedo = const_albedo)
                    # needs to travel next if alive

            # print('while niter2 = {}'.format(niter2))

        # if photon['is_alive'] and not photon['is_impacted']:
        if photon.is_alive and not photon.is_impacted:
            check_scattering(photon, forcepscatter=forcepscatter,
                             pscatterf=pscatterf, dfatm=dfatm)
            # need to travel next if scattered

        # if photon['is_alive']:
        if photon.is_alive:
            # print('S0 before', self.S0)
            # print('S1 before', self.S1)
            travel(photon, dfatm=dfatm)
            # print('S0 after', self.S0)
            # print('S1 after', self.S1)
            # print('travelling')
    # print('while niter = {}'.format(niter))
    return photon


# def plot_photon_path(self,
#               x, y, Z, hTOA, hMIN, cosz, phi,
#               alpha,
#               pscatter,
#               bound='periodic',  # or 'black'
#               launch = True):
#     # plot the path of a photon after
#     Y, X = np.meshgrid(y, x)
#
#     # COMPUTE TRIANGULATION FOR ENTIRE DOMAIN
#     restr = compute_triangles(
#         X, Y, Z, ccw=True, indx=None, indy=None, compute_areas=False)
#
#     TRIANGLES = restr['TRIANGLES']
#     TRI_POS = restr['TRI_POS']
#     POINTS = restr['POINTS']
#
#     if launch:
#         self.launch(x, y, X, Y, Z, hTOA, hMIN, cosz, phi,
#                  alpha, pscatter, bound=bound, track='true') # must track
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     tcf = ax.plot_trisurf(POINTS[:,0], POINTS[:,1], TRI_POS, POINTS[:,2],
#                           alpha = 0.5,  cmap='jet')
#     T0 =  self.TRACK0
#     T1 =  self.TRACK1
#     IMP = self.IMPTRACK
#     Tin = [a for a in self.INTRACK if a is not None]
#     Tout = [a for a in self.OUTTRACK if a is not None]
#
#     # plot trajectory
#     for t0, t1 in zip(T0, T1):
#         ax.plot([t0.x, t1.x], [t0.y, t1.y], [t0.z, t1.z], 'k')
#
#     if len(IMP) > 0: # plot poinTs of impact on the surface
#         for i in range(len(IMP)):
#            ax.scatter(IMP[i].x, IMP[i].y, IMP[i].z, color='red', marker = 'o')
#
#     if len(Tout) > 0: # plot exit points from boundaries
#         for i in range(len(Tout)):
#            ax.scatter(Tout[i].x, Tout[i].y, Tout[i].z, color='blue', marker = 'o')
#     if len(Tin) > 0: # plot entrance points from domain boundaries
#         for i in range(len(Tin)):
#            ax.scatter(Tin[i].x, Tin[i].y, Tin[i].z, color='orange', marker = 's')
#     ax.scatter(T1[-1].x, T1[-1].y, T1[-1].z, color='red', marker = '^')
#     ax.scatter(T0[0].x, T0[0].y, T0[0].z, color='cyan', marker = 'v')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     fig.colorbar(tcf, shrink=0.5, aspect=5)
#     plt.show()
#     return None



# def update_energy(photon, x, y, eTOA, eATM, eSRF,
#                   EDIR, EDIF, ERDIR, ERDIF, ECOUP):
#     eTOA += photon['Etoa']
#     eATM += photon['Eatm']
#     eSRF += photon['Esrf']
#     if photon['Esrf']:
#         i = np.argmin( np.abs(x-photon['S1'][0]))
#         j = np.argmin( np.abs(y-photon['S1'][1]))
#         if photon['direct'] and photon['nreflections'] == 0:
#             EDIR[i, j] += 1
#         elif photon['direct'] and photon['nreflections'] == 1:
#             ERDIR[i, j] += 1
#         elif not photon['direct'] and photon['nreflections'] == 0:
#             EDIF[i, j] += 1
#         elif not photon['direct'] and photon['nreflections'] == 1:
#             ERDIF[i, j] += 1
#         elif photon['nreflections'] > 1:
#             ECOUP[i, j] += 1
#         else:
#             Exception('update_energy: photon state not recognized!')
#     return eTOA, eATM, eSRF, EDIR, EDIF, ERDIR, ERDIF, ECOUP


@jit(nopython=True)
def update_energy(photon, myresult, entran):
    # update only arrays, do total counts at the end of simulation for each photon
    # if photon.Esrf:
    i = np.argmin( np.abs(myresult.x - photon.S1[0] ) )
    j = np.argmin( np.abs(myresult.y - photon.S1[1] ) )

    if photon.direct_hit and photon.direct and \
            not photon.reflected and not photon.multiple_refl:
        # myresult.EDIR[i, j] += entran
        myresult.EDIR[i, j] += types.float32(entran)

    elif photon.direct_hit and photon.direct and \
            photon.reflected and not photon.multiple_refl:
        # myresult.ERDIR[i, j] += entran
        myresult.ERDIR[i, j] += types.float32(entran)

    # right now the case of multiple scattering and single refl is here
    elif not photon.direct_hit and not photon.direct and \
            not photon.reflected and not photon.multiple_refl:
        # myresult.EDIF[i, j] += entran
        myresult.EDIF[i, j] += types.float32(entran)

    elif not photon.direct_hit and not photon.direct and \
         photon.reflected and not photon.multiple_refl:
        # myresult.ERDIF[i, j] += entran
        myresult.ERDIF[i, j] += types.float32(entran)

    # elif photon.multiple_refl or refl + subsequent scattering
    else:
        # myresult.ECOUP[i, j] += entran
        myresult.ECOUP[i, j] += types.float32(entran)
    # else:
    #     Exception('update_energy: photon state not recognized!')
    return myresult






# nphotons,
# eTOA, eABS, eSRF, EDIR, EDIF, ERDIR, ERDIF, ECOUP,
# x, y, X, Y, Z, hTOA, hMIN, cosz, phi,
# alpha_dir, alpha_dif, zelevmax,
# adaptive = adaptive, tilted = tilted, verbose = verbose,
# pscatterf = pscatterf, forcepscatter = forcepscatter,
# nb1 = nb1, nb2 = nb2, nb3 = nb3, dfatm = dfatm)

@jit(nopython=True)
def simulate( nphotons = None,
        x = None, y = None, Z = None,
              # hTOA = None, hMIN = None, zelevmax=None,
        cosz = None, phi = None,
        alpha_dir=None, alpha_dif=None,
        yalb=None, xalb=None, adirmat=None, adifmat=None,
        const_albedo=True, brdf = False,
        frac_energy=True, aerosol = True,
        adaptive=True, tilted=True,
        verbose = True, pscatterf=0, forcepscatter=False,
        nb1=10, nb2=6, nb3=6, dfatm=None
        ):


    ######################### CHECK DOMAIN CONSTRAINTS #############################
    zelevmax = np.max(Z) + 0.5
    zelevmin = np.min(Z)
    hTOA = dfatm['zz'][0] - 0.1  ###
    hMIN = dfatm['zz'][-1] - 100  ####
    assert hTOA < np.max(
        dfatm['zz']), 'Error: hTOA must be within atmospheric column'
    assert hMIN < np.min(
        dfatm['zz']), 'Error: hMIN must be within atmospheric column'
    assert hMIN < zelevmin, 'Error: hMIN must be lower than ground surface at all points'
    assert hTOA > zelevmax, "Error: hTOA must be larger than ground surface at all points"
    ################################################################################

    # initialize object with simulation results
    myresult = Result(x, y)

    for i in range(nphotons):
        # if i % 100 == 0:
            # print('Photon number = ', i)

        photon = launch(x=x, y=y, Z=Z, hTOA=hTOA, hMIN=hMIN,
                        cosz0=cosz, phi0=phi,
                        alpha_dir = alpha_dir, alpha_dif = alpha_dif,
                        adirmat = adirmat, adifmat = adifmat,
                        yalb = yalb, xalb = xalb, zelevmax = zelevmax,
                        const_albedo = const_albedo, brdf = brdf,
                        aerosol = aerosol,
                        frac_energy=frac_energy, myresult = myresult,
                        adaptive=adaptive, tilted=tilted, verbose = verbose,
                        pscatterf = pscatterf, forcepscatter = forcepscatter,
                        nb1=nb1, nb2=nb2, nb3=nb3, dfatm=dfatm)

        # update global energy counts
        myresult.eSRF = myresult.eSRF + photon.Esrf
        myresult.eABS = myresult.eABS + photon.Eatm
        myresult.eTOA = myresult.eTOA + photon.Etoa
        # print(photon.Esrf)
        # print(photon.Eatm)
        # print(photon.Etoa)
        # print(myresult.eSRF)
        # print(myresult.eABS)
        # print(myresult.eTOA)


        # eTOA, eABS, eSRF, EDIR, EDIF, ERDIR, ERDIF, ECOUP = update_energy(
        #     photon, x, y, eTOA, eABS, eSRF, EDIR, EDIF, ERDIR, ERDIF, ECOUP)

    # return eTOA, eABS, eSRF, EDIR, EDIF, ERDIR, ERDIF, ECOUP
    return myresult


################################################################################

########################     END PHOTON FUNCTIONS      #########################

################################################################################



# CHECK WHETHER SEGMENTS AB and CD INTERSECT EACH OTHER
# ONLY INVOLVES x and y dimensions of the points


@jit(nopython=True)
def myatan2(x, y):
    # atan2 function adapted for my coordinate system
    # x, y in correct order, add 2pi if negative to stay in [0, 2pi]
    # to check::
    # print('checking for my system of coords::')
    # print('x=0, y=1: ',  np.arctan2(0, 1)/np.pi)
    # print('x=1, y=0: ',  np.arctan2(1, 0)/np.pi)
    # print('x=0, y=-1: ', np.arctan2(0, -1)/np.pi)
    # print('x=-1, y=0: ', 2 + np.arctan2(-1, 0)/np.pi)
    # print('x=-1, y=0: ', 2 + np.arctan2(-0.01, -1)/np.pi)
    # print('x=-0.01, y=1: ', 2 + np.arctan2(-0.5, -0.5)/np.pi)
    atan2 = np.arctan2(x, y)
    if atan2 < 0.0:
        atan2 = 2.0 * np.pi + atan2
    return atan2


@jit(nopython=True)
def twonormal(v):
    # get two vectors orthogonal to v
    # rhs triplets of normalized vectors
    vn = v/np.linalg.norm(v)
    # randv = np.random.rand(3)
    randv = np.array([random_num(), random_num(), random_num()])
    t1 = randv - randv.dot(vn) * vn
    t1n = t1/np.linalg.norm(t1)
    t2n = np.cross(vn, t1n)
    return vn, t1n, t2n

# vn, t1, t2 = twonormal([1,0,0])

@jit(nopython=True)
def polar2cart(r=1.0, cosz=1.0, phi = 0.0, x0vec = (0.0,0.0,0.0)):
    # using photomc conversion, by default get unit vector for r=1:
    XV = np.zeros(3)
    XV[2] = x0vec[2] + r*cosz
    XV[1] = x0vec[1] + r*np.sqrt(1.0-cosz**2)*np.cos(phi)
    XV[0] = x0vec[0] + r*np.sqrt(1.0-cosz**2)*np.sin(phi)
    # y = x0vec[1] + r*np.sin(np.arccos(cosz))*np.cos(phi)
    # x = x0vec[2] + r*np.sin(np.arccos(cosz))*np.sin(phi)
    return XV


@jit(nopython=True)
def cart2polar(x=0.0, y=0.0, z=0.0):
    r = np.sqrt(x**2 + y**2 + z**2)
    cosz = z/r
    phi = myatan2(x, y)
    # phi = np.arcsin(x/r/np.sqrt(1-cosz**2))
    # phi = np.arcsin(x/r/np.sin(np.arccos(cosz)))
    return r, cosz, phi


@jit(nopython=True)
def point_distance(P1, P2):
    # compute the distance between 2 points in 3D
    # P1, P2 must arrays or lists of length 3
    dist = ( (P1[0] - P2[0])**2 + (P1[1] - P2[1])**2 + (P1[2] - P2[2])**2 )**0.5
    return dist


@jit(nopython=True)
def area_triangle(P1, P2, P3):
    # P1, P2, P3 -> 3 points, triangle vertices
    V1 = P2 - P1
    V2 = P3 - P1
    normV1 = np.sqrt(np.dot(V1, V1))
    normV2 = np.sqrt(np.dot(V2, V2))
    costheta = np.dot(V1, V2)/normV1/normV2
    # area = 0.5*normV1*normV2*np.sin(np.arccos(costheta))
    area = 0.5 * normV1 * normV2 * np.sqrt( 1.0 - costheta**2 )
    # area = 0.5*np.linalg.norm(np.cross(V1, V2))
    return area


@jit(nopython=True)
def ccw(A, B, C):
    # return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


@jit(nopython=True)
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


@jit(nb.types.Tuple(   ( nb.boolean, nb.float64[:], nb.float64, nb.float64[:]) ) (
    nb.float64[:], nb.float64[:], nb.float64[:],nb.float64[:], nb.float64[:]),
    nopython=True)
def check_triang_intersect(p1, p2, p3, s1, s2):
    '''--------------------------------------------------------------------------
      check whether a triangle with vertices (p1, p2, p3)
      intersects the segment with end points (s1, s2
      return a boolean True is intersection.
      input must be 3D numpy arrays

    NOTES:
    # COMPUTE THE TRIANGLE NORMAL (ORIENTED):
    N = np.cross(p2-p1, p3-p1)

    # NORMAL VECTORS FOR EACH TRIANGLE SIDE (ORIENTED OUTWARD):
    N12 = (p2 - p1)*N
    N23 = (p3 - p2)*N
    N31 = (p1 - p3)*N

    # the line segment between points s1 and s2 is
    # R(t) = s1 + t (s2 - s1) with t in [0, 1]

    # The minimum distance between a point p in the triangle
    # and the line passing through p1 and p2 is:
    # Dist = ((p - p1) • N12) / |N12|
    # with |N12| being the norm of the side normal and • the dot product
    # Dist is positive if point is outside the triangle
    # if all distances between p and the three points are < 0, point is inside

    #The triangle's plane is defined by the unit normal N and the distance
    # to the origin D. So the plane equation is N • x + D = 0
    # where x is a 3D point,
    # The distance to the origin D can be computing using any point of the
    # triangle, for example D = -(N • p1)

    # The intersection of line segment R(t) and the plane happens when t is
    # t = - (D + N • s1) / (N • (s2 - s1))
    --------------------------------------------------------------------------'''

    # if point_type:
    #     p1 = np.array([p1.x, p1.y, p1.z])
    #     p2 = np.array([p2.x, p2.y, p2.z])
    #     p3 = np.array([p3.x, p3.y, p3.z])
    #     s1 = np.array([s1.x, s1.y, s1.z])
    #     s2 = np.array([s2.x, s2.y, s2.z])
    # USE POINTS ONLY NOW


       # print('DIAGNOSTICS')
       # print(p1)
       # print(p2)
       # print(p3)

    # COMPUTE THE TRIANGLE NORMAL (ORIENTED):
    N = np.cross(p2 - p1, p3 - p1)

    # myN = np.array([0.0, 0.0, 0.0]) # upward direction
    # for ix3 in range(3):
    #     myN[ix3] = N[ix3]
    # print('N=', N)

    # NORMAL VECTORS FOR EACH TRIANGLE SIDE (ORIENTED OUTWARD):
    # N12 = (p2 - p1) * N
    # N23 = (p3 - p2) * N
    # N31 = (p1 - p3) * N

    # D = - np.dot(N, p1) # distance to the origin
    D = - mydot(N, p1, n=3) # distance to the origin

    # print('D=', D)

    # if t is between 0 and 1, the intersection point is in the segment
    numer = - (D + mydot(N, s1, n=3))
    # numer = - (D + np.dot(N, s1))
    # denom = np.dot(N, (s2 - s1))
    denom = mydot(N, (s2 - s1), n=3)
    if np.abs(denom)> 1E-9:
        t = numer / denom
    else:
        t = 1E9

    # we have computed the intersection point between line and plane
    # now check that the point is between the endpoints of the segment
    # inter_segment = np.logical_and(t>0, t<1)
    is_intersect = np.logical_and(t > 0, t < 1)

    # now check that the intersection point is within the triangle
    # Area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
    # s = 1 / (2 * Area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
    # t = 1 / (2 * Area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)

    # Area = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
    # s = 1 / (2 * Area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
    # t = 1 / (2 * Area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)

    # inter_triangle = s>0 and t>0 and 1-s-t>0


    # is_intersect = np.logical_and(inter_triangle, inter_segment)
    # COMPUTE INTERSECTION POINT:
    intp = np.zeros(3)
    if is_intersect:
        intp = s1 + t * (s2 - s1)

    # if point_type and is_intersect:
    #     intp = Point(intp[0], intp[1], intp[2])

    return is_intersect, intp, t, N

@jit(nopython=True)
def segment_bounding_box(S1, S2, x, y):
    """---------------
    Compute a 3D bounding box around a segment between points S1 and S2
    S1, S2 must be point object, or 3d array-like or list-like objects
    # x, y -> array of point coords along two coord directions.
    ----------------"""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    eps = 1E-6
    # if not isinstance(S1, Point):
    #     S1 = Point(S1[0], S1[1], S1[2])
    # if not isinstance(S2, Point):
    #     S2 = Point(S2[0], S2[1], S2[2])
    # segbox = {}
    # segbox['xmin'] = min(S1.x, S2.x)
    # segbox['xmax'] = max(S1.x, S2.x)
    # segbox['ymin'] = min(S1.y, S2.y)
    # segbox['ymax'] = max(S1.y, S2.y)
    # segbox['zmin'] = min(S1.z, S2.z)
    # segbox['zmax'] = max(S1.z, S2.z)

    # segbox = {}
    # segbox['xmin'] = min(S1[0], S2[0])
    # segbox['xmax'] = max(S1[0], S2[0])
    # segbox['ymin'] = min(S1[1], S2[1])
    # segbox['ymax'] = max(S1[1], S2[1])
    # segbox['zmin'] = min(S1[2], S2[2])
    # segbox['zmax'] = max(S1[2], S2[2])
    # segbox['indx'] = np.where(np.logical_and(x > segbox['xmin'] - dx - eps,
    #                                          x < segbox['xmax'] + dx + eps))[0]
    # segbox['indy'] = np.where(np.logical_and(y > segbox['ymin'] - dy - eps,
    #                                          y < segbox['ymax'] + dy + eps))[0]

    segbox_xmin = min(S1[0], S2[0])
    segbox_xmax = max(S1[0], S2[0])
    segbox_ymin = min(S1[1], S2[1])
    segbox_ymax = max(S1[1], S2[1])
    # segbox_zmin = min(S1[2], S2[2])
    # segbox_zmax = max(S1[2], S2[2])

    indx = np.where(np.logical_and(x > segbox_xmin - dx - eps,
                                             x < segbox_xmax + dx + eps))[0]
    indy = np.where(np.logical_and(y > segbox_ymin - dy - eps,
                                             y < segbox_ymax + dy + eps))[0]

    return indx, indy



# @jit(nopython=True)
@jit(nopython=True)
def compute_triangles(x, y, Z, ccw=True, indx=None, indy=None):
    # if specific indices are not provided, use all domain:
    if indx is None:
        nx = np.shape(Z)[0]
        indx = np.arange(nx)
    else:
        nx = np.shape(indx)[0]
        # nx = np.shape(indx)[0]
    if indy is None:
        ny = np.shape(Z)[1]
        indy = np.arange(ny)
    else:
        ny = np.shape(indy)[0]
        # ny = np.size(indy)

    # print(nx)

    # store all points column-wise
    # POINTS = np.zeros((3, nx*ny))
    POINTS = np.zeros((nx*ny, 3))
    for j, indj in enumerate(indy):
        for i, indi in enumerate(indx):
            count = j * nx + i
            # POINTS[count, 0] = X[indi, indj]
            # POINTS[count, 1] = Y[indi, indj]
            # POINTS[count, 2] = Z[indi, indj]
            POINTS[count, 0] = x[indi]
            POINTS[count, 1] = y[indj]
            POINTS[count, 2] = Z[indi, indj]

    # if not (nx > 1 and ny > 1):
    #    print('Error!')
    #    pass

    # if compute_areas:
    #     print('Compute Areas function not yet available')
    #     AREAS = np.zeros((nx, ny))

    count = 0
    # TRIANGLES = []
    # TRI_POS = np.zeros((2 * (nx - 1) * (ny - 1), 3), dtype=np.int32)

    # TRI_POS = np.zeros((2 * (nx - 1) * (ny - 1), 3), dtype=np.int32)
    # TP1 =     np.zeros((2 * (nx - 1) * (ny - 1), 3), dtype=np.float64)
    # TP2 =     np.zeros((2 * (nx - 1) * (ny - 1), 3), dtype=np.float64)
    # TP3 =     np.zeros((2 * (nx - 1) * (ny - 1), 3), dtype=np.float64)

    TRI_POS = np.zeros((3, 2 * (nx - 1) * (ny - 1)), dtype=np.int32)
    TP1 =     np.zeros((3, 2 * (nx - 1) * (ny - 1)), dtype=np.float64)
    TP2 =     np.zeros((3, 2 * (nx - 1) * (ny - 1)), dtype=np.float64)
    TP3 =     np.zeros((3, 2 * (nx - 1) * (ny - 1)), dtype=np.float64)

    for i in indx[1:]:
        for j in indy[1:]:

            # PNE = Point(X[i - 1, j - 1], Y[i - 1, j - 1], Z[i - 1, j - 1])
            # PSE = Point(X[i, j - 1], Y[i, j - 1], Z[i, j - 1])
            # PNW = Point(X[i - 1, j], Y[i - 1, j], Z[i - 1, j])
            # PSW = Point(X[i, j], Y[i, j], Z[i, j])


            # PNE = np.array([X[i - 1, j - 1], Y[i - 1, j - 1], Z[i - 1, j - 1]])
            # PSE = np.array([X[i, j - 1],     Y[i, j - 1],     Z[i, j - 1]])
            # PNW = np.array([X[i - 1, j],     Y[i - 1, j],     Z[i - 1, j]])
            # PSW = np.array([X[i, j],         Y[i, j],         Z[i, j] ])

            PNE = np.array([x[i - 1],     y[ j - 1],     Z[i - 1, j - 1]])
            PSE = np.array([x[i],         y[ j - 1],     Z[i, j - 1]])
            PNW = np.array([x[i - 1],     y[j],          Z[i - 1, j]])
            PSW = np.array([x[i],         y[j],          Z[i, j] ])

            PNE_IND = np.array( [ i - 1, j - 1  ], dtype=np.int32)
            PNW_IND = np.array( [ i - 1, j      ], dtype=np.int32)
            PSE_IND = np.array( [ i,     j - 1  ], dtype=np.int32)
            PSW_IND = np.array( [ i,     j      ], dtype=np.int32)

            if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
               if ccw:
                   # t1 = Triangle(PSE, PNW, PNE)
                   # t2 = Triangle(PSE, PSW, PNW)
                   t1 =  {'P1':  PSE,        'P2':  PNW,        'P3':  PNE}
                   t1_ind = {'P1ix':PSE_IND[0], 'P2ix':PNW_IND[0], 'P3ix':PNE_IND[0],
                          'P1jy':PSE_IND[1], 'P2jy':PNW_IND[1], 'P3jy':PNE_IND[1]}
                   t2 =  {'P1':  PSE,        'P2':  PSW,        'P3':  PNW}
                   t2_ind = {       'P1ix':PSE_IND[0], 'P2ix':PSW_IND[0], 'P3ix':PNW_IND[0],
                          'P1jy':PSE_IND[1], 'P2jy':PSW_IND[1], 'P3jy':PNW_IND[1]}
               else:
                   t1 = {'P1':PNE, 'P2':PNW, 'P3':PSE}
                   t1_ind={      'P1ix':PNE_IND[0], 'P2ix':PNW_IND[0], 'P3ix':PSE_IND[0],
                         'P1jy':PNE_IND[1], 'P2jy':PNW_IND[1], 'P3jy':PSE_IND[1]}
                   t2 = {'P1':PNW, 'P2':PSW, 'P3':PSE}
                   t2_ind = {      'P1ix': PNW_IND[0], 'P2ix': PSW_IND[0], 'P3ix': PSE_IND[0],
                         'P1jy': PNW_IND[1], 'P2jy': PSW_IND[1], 'P3jy': PSE_IND[1]}
            # one even and one odd
            # elif (i % 2 == 0 and j % 2 != 0) or (i % 2 != 0 and j % 2 == 0):
            else:
               if ccw:
                   t1 = {'P1':PSW,          'P2':PNW,          'P3':PNE}
                   t1_ind = {      'P1ix':PSW_IND[0], 'P2ix':PNW_IND[0], 'P3ix':PNE_IND[0],
                         'P1jy':PSW_IND[1], 'P2jy':PNW_IND[1], 'P3jy':PNE_IND[1]}
                   t2 = {'P1':PSE,           'P2':PSW,          'P3':PNE}
                   t2_ind = {      'P1ix':PSE_IND[0], 'P2ix':PSW_IND[0], 'P3ix':PNE_IND[0],
                         'P1jy':PSE_IND[1], 'P2jy':PSW_IND[1], 'P3jy':PNE_IND[1]}
               else:
                   t1 = {'P1':PNE,          'P2':PNW,          'P3':PSW}
                   t1_ind = {      'P1ix':PNE_IND[0], 'P2ix':PNW_IND[0], 'P3ix':PSW_IND[0],
                         'P1jx':PNE_IND[1], 'P2jy':PNW_IND[1], 'P3jy':PSW_IND[1]}
                   t2 = {'P1':PNE,           'P2':PSW,             'P3':PSE}
                   t2_ind = {      'P1ix': PNE_IND[0], 'P2ix': PSW_IND[0], 'P3ix': PSE_IND[0],
                         'P1jy': PNE_IND[1], 'P2jy': PSW_IND[1], 'P3jy': PSE_IND[1]}

            # save the three points of the two triangles::
            # if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
            #     if ccw:
            #         TP1[count, :] = PSE  # triangle 1 point 1
            #         TP2[count, :] = PNW
            #         TP3[count, :] = PNE
            #         TP1[count+1, :] = PSE  # triangle 1 point 1
            #         TP2[count+1, :] = PSW
            #         TP3[count+1, :] = PNW
            #     else:
            #         TP1[count, :] = PNE  # triangle 1 point 1
            #         TP2[count, :] = PNW
            #         TP3[count, :] = PSE
            #         TP1[count+1, :] = PNW  # triangle 1 point 1
            #         TP2[count+1, :] = PSW
            #         TP3[count+1, :] = PSE
            # else:
            #     if ccw:
            #         TP1[count, :] = PSW  # triangle 1 point 1
            #         TP2[count, :] = PNW
            #         TP3[count, :] = PNE
            #         TP1[count+1, :] = PSE  # triangle 1 point 1
            #         TP2[count+1, :] = PSW
            #         TP3[count+1, :] = PNE
            #     else:
            #         TP1[count, :] = PNE  # triangle 1 point 1
            #         TP2[count, :] = PNW
            #         TP3[count, :] = PSW
            #         TP1[count+1, :] = PNE  # triangle 1 point 1
            #         TP2[count+1, :] = PSW
            #         TP3[count+1, :] = PSE

            if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                if ccw:
                    TP1[:, count] = PSE  # triangle 1 point 1
                    TP2[:, count] = PNW
                    TP3[:, count] = PNE
                    TP1[:, count+1] = PSE  # triangle 1 point 1
                    TP2[:, count+1] = PSW
                    TP3[:, count+1] = PNW
                else:
                    TP1[:, count] = PNE  # triangle 1 point 1
                    TP2[:, count] = PNW
                    TP3[:, count] = PSE
                    TP1[:, count+1] = PNW  # triangle 1 point 1
                    TP2[:, count+1] = PSW
                    TP3[:, count+1] = PSE
            else:
                if ccw:
                    TP1[:, count] = PSW  # triangle 1 point 1
                    TP2[:, count] = PNW
                    TP3[:, count] = PNE
                    TP1[:, count+1] = PSE  # triangle 1 point 1
                    TP2[:, count+1] = PSW
                    TP3[:, count+1] = PNE
                else:
                    TP1[:, count] = PNE  # triangle 1 point 1
                    TP2[:, count] = PNW
                    TP3[:, count] = PSW
                    TP1[:, count+1] = PNE  # triangle 1 point 1
                    TP2[:, count+1] = PSW
                    TP3[:, count+1] = PSE


            # if compute_areas:
            #     # AREAS[i,j] += t1.area()
            #     # AREAS[i,j] += t2.area()
            #     AREAS[i,j] += area_triangle(t1['P1'], t1['P2'], t1['P3'])
            #     AREAS[i,j] += area_triangle(t2['P1'], t2['P2'], t2['P3'])


            # t1.set_node_ind(t1.P1.jy * nx + t1.P1.ix,
            #                 t1.P2.jy * nx + t1.P2.ix,
            #                 t1.P3.jy * nx + t1.P3.ix)
            #
            # t2.set_node_ind(t2.P1.jy * nx + t2.P1.ix,
            #                 t2.P2.jy * nx + t2.P2.ix,
            #                 t2.P3.jy * nx + t2.P3.ix)

            # TRI_POS[count, 0] = t1_ind['P1jy'] * nx + t1_ind['P1ix']
            # TRI_POS[count, 1] = t1_ind['P2jy'] * nx + t1_ind['P2ix']
            # TRI_POS[count, 2] = t1_ind['P3jy'] * nx + t1_ind['P3ix']
            #
            # TRI_POS[count + 1, 0] = t2_ind['P1jy'] * nx + t2_ind['P1ix']
            # TRI_POS[count + 1, 1] = t2_ind['P2jy'] * nx + t2_ind['P2ix']
            # TRI_POS[count + 1, 2] = t2_ind['P3jy'] * nx + t2_ind['P3ix']


            TRI_POS[0, count] = t1_ind['P1jy'] * nx + t1_ind['P1ix']
            TRI_POS[1, count] = t1_ind['P2jy'] * nx + t1_ind['P2ix']
            TRI_POS[2, count] = t1_ind['P3jy'] * nx + t1_ind['P3ix']

            TRI_POS[0, count + 1] = t2_ind['P1jy'] * nx + t2_ind['P1ix']
            TRI_POS[1, count + 1] = t2_ind['P2jy'] * nx + t2_ind['P2ix']
            TRI_POS[2, count + 1] = t2_ind['P3jy'] * nx + t2_ind['P3ix']

            count += 2
            # TRIANGLES.append(t1)
            # TRIANGLES.append(t2)

    # now recompute the area as the average of the 4 neighbour cells
    # AREAS2 = np.zeros((nx, ny))
    # AREAS2[0,0] = AREAS[1,1]
    # AREAS2[nx-1,0] = AREAS[nx-1,1]
    # AREAS2[0, ny-1] = AREAS[1, ny-1]
    # AREAS2[nx-1, ny-1] = AREAS[nx-1, ny-1]
    # for i in range(nx):
    #     for j in range(ny):
    #         if i == 0 and j == 0:
    #
    # AREAS2[i,j] =

    # RES = {}
    # RES['TRIANGLES'] = TRIANGLES
    # RES['TRI_POS'] = TRI_POS
    # RES['POINTS'] = POINTS

    # RES = {'TRIANGLES':TRIANGLES, 'TRI_POS': TRI_POS, 'POINTS':POINTS}
    # if compute_areas:
    #     RES['AREA'] = AREAS

    return TP1, TP2, TP3, TRI_POS, POINTS


# def triang_intersect_2d(S1, S2, TRIANGLES, TRI_POS):
#     # keep only the triangles in the list for which
#     # their horizontal projection intersects the horizontal proj.
#     # of the segment S1, S2
#     # return the list of triangles kept
#     # intersection -> at least one of the three adges intersects the segment
#     # NOT RECOMMENDED - DO NOT USE IN 3D APPLICATIONS, ONLY 2D
#     ntriang = len(TRIANGLES)
#     INTER = np.zeros((3, ntriang)).astype(bool)
#     for it, tri in enumerate(TRIANGLES):
#         # print(tri)
#         INTER[0, it] = intersect(S1, S2, tri.P1, tri.P2)
#         INTER[1, it] = intersect(S1, S2, tri.P2, tri.P3)
#         INTER[2, it] = intersect(S1, S2, tri.P3, tri.P1)
#     INTERT = np.amax(INTER, axis=0)
#     KEPTR = [a for ia, a in enumerate(TRIANGLES) if INTERT[ia]]
#     KEPTPOS = [a for ia, a in enumerate(TRI_POS) if INTERT[ia]]
#     return KEPTR, KEPTPOS



# @jit(nb.types.Tuple(   ( nb.boolean, nb.float64[:], nb.float64, nb.float64[:]) ) (
#     nb.float64[:], nb.float64[:], nb.float64[:],nb.float64[:], nb.float64[:], nb.float64[:,:]),
#     nopython=True)

@jit(nopython=True)
def find_first_3d_intersection(S1, S2, TP1, TP2, TP3, TRI_POS):
    # ntriang = len(TP1)
    # ntriang = TP1.shape[0]
    ntriang = TP1.shape[1]
    INTERPOINTS = []
    # INTERTRIANGLES = []
    INTERPOS = []
    # INTERC = np.zeros(ntriang, dtype=np.bool)
    # INTERC = np.broadcast_to(False, ntriang)
    INTERC = np.repeat(False, ntriang)
    NORMALS = []
    # for it, tri in enumerate(TRIANGLES):
    for it in range(ntriang):
        # print(tri)

        # print(S1.distance(tri.P1))
        # print(S1.distance(tri.P2))
        # print(S1.distance(tri.P3))

        # this find if segment intercept plane, no within trinagle
        # is_intersect, intp, t, N
        # inters = check_triang_intersect(tri.P1, tri.P2, tri.P3,
        #                                    S1, S2, point_type=True)

        # is_intersect, intp, t, mynormal = check_triang_intersect(
        #                                 tri.P1, tri.P2, tri.P3,
        #                                 S1, S2, point_type=True)

        # is_intersect, intp, t, mynormal = check_triang_intersect(
        #     tri['P1'], tri['P2'], tri['P3'],
        #     S1, S2, point_type=True)

        is_intersect, intp, t, mynormal = check_triang_intersect(
            # TP1[it, :], TP2[it, :], TP3[it, :],
            TP1[:, it], TP2[:, it], TP3[:, it],
            S1, S2)
        # print(inters)
        # print(inters)
        # INTERC[it] = inters[0]


        INTERC[it] = is_intersect

        # COMPUTE THE INTERSECTION POINT:
        if is_intersect:
            # A = np.column_stack( (tri['P3'] - tri['P1'],
            #                      tri['P2'] - tri['P1'],
            #                      S1 - S2))

            # A = np.column_stack( (TP3[it, :] - TP1[it, :],
            #                       TP2[it, :] - TP1[it, :],
            #                       S1 - S2))

            A = np.column_stack( (TP3[:, it] - TP1[:, it],
                                  TP2[:, it] - TP1[:, it],
                                  S1 - S2))

            # b = S1 - TP1[it, :]
            b = S1 - TP1[:, it]

            params = np.linalg.inv(A).dot(b)
            intpv = S1 - params[2] * (S1 - S2)

            # check whether point belong to triangle:
            is_in_triangle = params[0] > 0 and params[1] > 0 and params[0] + params[1] < 1

            if is_in_triangle:
                INTERPOINTS.append( np.array([intpv[0], intpv[1], intpv[2]]))
                # INTERTRIANGLES.append(tri)
                # INTERPOS.append(TRI_POS[it, :])
                INTERPOS.append(TRI_POS[:, it])
                NORMALS.append(mynormal)

    # init results

    if len(INTERPOINTS) == 0:
        IS_INT = False
        MYNORM = np.array([-9999.9, -9999.9, -9999.9])
        INTPOINT = np.array([-9999.9, -9999.9, -9999.9])
    else:
        # DISTANCES = [S2.distance(other) for other in INTERPOINTS]
        # DISTANCES = [S1.distance(other) for other in INTERPOINTS]
        lenp = len(INTERPOINTS)
        DISTANCES = np.zeros(lenp)
        for elemi in range(lenp):
            DISTANCES[elemi] = point_distance(S1, INTERPOINTS[elemi])
        mindist_indx = np.int(np.argmin(DISTANCES))
        # print('index of min distance', mindist_indx)
        INTPOINT = INTERPOINTS[mindist_indx]
        # MYTR = INTERTRIANGLES[mindist_indx]
        # MYPOS = [INTERPOS[mindist_indx]]
        MYNORM = NORMALS[mindist_indx]
        IS_INT = True
        # res['is_intersected'] = True
        # res['int_point'] = intpoint
        # res['distances'] = DISTANCES
        # res['MYTR'] = MYTR
        # res['MYPOS'] = MYPOS
        # res['MYNORM'] = MYNORM


    # res = {'is_intersected':is_intersected,
    #        'int_point': intpoint,
    #        'MY_NORM': MYNORM}

    return IS_INT, MYNORM, INTPOINT


@jit(nopython=True)
def check_surface_intersections(S1, S2, x, y, Z,
                                usebox=True, check_2d=False):

    # segbox = segment_bounding_box(S1, S2, x, y)
    segbox_indx, segbox_indy = segment_bounding_box(S1, S2, x, y)
    # print('SEGBOX')
    # print('points = {} {}'.format(S1, S2))
    # print(segbox)
    if not usebox:
        # build triangle mesh for entire domain
        # print('len(rest) = ', len(rest))
        TP1, TP2, TP3, TRI_POS, POINTS = compute_triangles(
              x, y, Z, ccw=True,
              indx=np.arange(np.shape(x)[0]),
              indy=np.arange(np.shape(y)[0]))
             # compute_areas=False)
    else:
        # build triangle mesh only in the vicinity of the segment S1-S2
        triangles = compute_triangles(x, y, Z, ccw=True, indx=segbox_indx,
                                      indy=segbox_indy)
        TP1, TP2, TP3, TRI_POS, POINTS = triangles
             # compute_areas=False)

    # TRIANGLES = rest['TRIANGLES']
    # TRI_POS = rest['TRI_POS']
    # POINTS = rest['POINTS']

    # print('ntriangles = {}'.format(len(TRIANGLES)))

    # filter triangles based on intersections on the horizontal plane
    # to speed up the search for the intersection point
    # if check_2d:
    #     print('WARNING: DOING THE 2D CHECK!!!')
    #     KEPT_TRI, KEPT_POS = triang_intersect_2d(S1, S2, TRIANGLES, TRI_POS)
    # else:
    # KEPT_TRI = TRIANGLES
    # KEPT_POS = TRI_POS

    # if len(TP1) > 0:
    if np.shape(TP1)[1] > 0:
        intersection = find_first_3d_intersection(S1, S2, TP1, TP2, TP3,
                                                  TRI_POS)
        IS_INT, MYNORM, INTPOINT = intersection
    else:
        # res = {}
        # res = {'is_intersected':False}
        IS_INT = False
        MYNORM = np.array([-9999.9, -9999.9, -9999.9])
        INTPOINT = np.array([-9999.9, -9999.9, -9999.9])

        # res = {'is_intersected':False,
        #        'int_point': np.array([-9999.9, -9999.9, -9999.9]),
        #        'MY_NORM': np.array([-9999.9, -9999.9, -9999.9])}
    return IS_INT, MYNORM, INTPOINT


@jit(nopython=True)
def intersect_plane(S1, S2, P1, P2, P3):
    # find the point of intersection between segment and plane
    # segment S1 - S2
    # plane containing the point P1 - P2 - P3

    # if not isinstance(S1, Point):
    #     S1 = Point(S1[0], S1[1], S1[2])
    # if not isinstance(S2, Point):
    #     S2 = Point(S2[0], S2[1], S2[2])
    #
    # if not isinstance(P1, Point):
    #     S1 = Point(P1[0], P1[1], P1[2])
    # if not isinstance(P2, Point):
    #     S2 = Point(P2[0], P2[1], P2[2])
    # if not isinstance(P3, Point):
    #     S2 = Point(P3[0], P3[1], P3[2])

    # This also work for a plane, not necessarily within the triangle
    is_intersect, intp, t, mynormal = check_triang_intersect(
          P1, P2, P3, S1, S2)

    intpv = None
    if is_intersect:
        A = np.column_stack((P3 - P1,  P2 - P1,  S1 - S2))
        b = S1 - P1
        params = np.linalg.inv(A).dot(b)
        intpv = S1 - params[2] * (S1 - S2)

        # intpv = (intpv[0], intpv[1], intpv[2])

    return is_intersect, intpv


@jit(nopython=True)
def running_max(x):
    # compute running maxima of size (3):
    nx = np.size(x)
    runx = np.zeros(nx)
    for i in range(nx):
        if i < 2:
            runx[i] = np.max(x[:2])
        elif i > nx - 2:
            runx[i] = np.max(x[nx - 2:])
        else:
            runx[i] = np.max(x[i - 1:i + 2])
    return runx


@jit(nopython=True)
def path_absorption(photon, dfatm=None):
    # print('path absorption::')
    # print(photon.S0)
    # print(photon.S1)
    z0 = photon.S0[2]
    z1 = photon.S1[2]
    cosz = photon.cosz
    # print('cosz =', cosz)

    zup = max(z0, z1)
    zdown = min(z0, z1)


    ztemp = zup
    # ztemp = zup

    # print("*****************")
    # print('aerosol =', photon.aerosol)
    # print("*****************")

    if photon.aerosol:
        absparam = 'k_abs_tot'
    else:
        absparam = 'k_abs_gas'

    zlower = get_atm_value(ztemp, dfatm=dfatm, param='zz')
    mybeta_abs = get_atm_value(ztemp, dfatm=dfatm, param=absparam)

    # print('mybeta_abs', mybeta_abs)
    # print('z0', z0)
    # print('z1', z1)

    while zlower > zdown:

        dz = ztemp - zlower
        dl = np.abs(dz/cosz)
        energy_kept_fraction = np.exp( - dl * mybeta_abs)

        photon.Eatm = photon.Eatm + photon.energy * (1 - energy_kept_fraction)
        photon.energy = photon.energy * (energy_kept_fraction)

        ztemp = zlower - 1E-9 # to make sure it stays below the level
        zlower = get_atm_value(ztemp, dfatm=dfatm, param='zz')
        mybeta_abs = get_atm_value(ztemp, dfatm=dfatm, param=absparam)

    # at last complete the last step:
    dz = ztemp - zdown
    dl = np.abs(dz / cosz)
    energy_kept_fraction = np.exp(- dl * mybeta_abs)
    photon.Eatm = photon.Eatm + photon.energy * (1 - energy_kept_fraction)
    photon.energy = photon.energy * (energy_kept_fraction)

    # print('Eatm', photon.Eatm)
    # print('energy', photon.energy)

    return photon


@jit(nopython=True)
def compute_travel_distance(photon, dfatm=None):
    '''-------------------------------------------------------------------------
    Computes the path travelled by a random photon before it is
    absorbed or scattered
    INPUT:*
    - z0 -> initial photon height
    - cosz -> cosine of the zenith angle (negative for downward photon!!)
    - dfatm -> data frame containing the information about the atmosphere
    OUTPUT::
    - dl -> length travelled by the photon (magnitude only, no sign)
    - new_z -> new elevation at the end of the photon's run
    - impact -> boolean, if True the photon hits the lower domain boundary
            (the bottom of the prescribed atmospheric column)
    -escaped -> boolean, True if the photon escapes from the top of atmosphere
    - taul -> prescribed optical depth of medium (from randomly generated xi)
    - tautemp -> optical depth optained fron integration along the path
                 (it should be equal to taul if convergence is reached)
    -------------------------------------------------------------------------'''
    # taul = - np.log(random.random())  # generate De Beer's Law optical depth
    taul = - np.log(random_num())  # generate De Beer's Law optical depth
    tautemp = 0  # init numerical optical depth value
    converged = False  # init
    cosz = photon.cosz
    z0 = photon.S0[2]
    new_z = z0  # init final elevation values
    dztot = 0  # init total vertical distance travelled
    impact = False  # init - becomes true if photon hits the lower domain bound
    escaped = False  # init - becomes true if photon leaves the atm. from above

    # iterate until reached convergence, or photon leaves the domain:
    zz = dfatm['zz']
    while not converged and new_z > np.min(zz) and new_z < np.max(zz):
        # compute current extinction coefficient beta (based on current z val)
        # FRACEN
        # mybeta = get_atm_value(new_z, dfatm=dfatm, param='extb') # IF NOT FRAC
        # mybeta_abs = get_atm_value(new_z, dfatm=dfatm, param='absb') # IF FRAC ENER

        if photon.aerosol:
            ext_coeff = 'k_ext_tot' # total coefficient (gas + aerosol)
            sca_coeff = 'k_sca_tot' # total coefficient (gas + aerosol)
        else:
            ext_coeff = 'k_ext_gas' # gas only scattering coefficient
            sca_coeff = 'k_sca_gas' # gas only scattering coefficient

        if photon.frac_energy:
            mybeta = get_atm_value(new_z, dfatm=dfatm, param=sca_coeff)
        else:
            mybeta = get_atm_value(new_z, dfatm=dfatm, param=ext_coeff)

        # mylevel = get_atm_value(new_z, dfatm=dfatm, param='levels')
        # print('mylevel = {}'.format(mylevel))
        # if photon.verbose:
        # print('compute travel distance:')
        # mylevel = get_atm_value(new_z, dfatm=dfatm, param='levels')
        # print('mylevel = {}', mylevel)
        # print('my ext coeff =', mybeta)
        # compute thickness of atm. cell, and its lower boudnary elevation
        mydz = get_atm_value(new_z, dfatm=dfatm, param='dz')
        lowerz = get_atm_value(new_z, dfatm=dfatm, param='zz')
        dz_down = lowerz - new_z  # distance from bound below (negative quantity)
        dz_up = lowerz + mydz - new_z  # distance from bound above (positive)
        # print(lowerz, dz_down, dz_up)
        if cosz > 0:
            dz_tobound = dz_up  # going up; distance to upper level
            eps = 1E-6
        else:
            dz_tobound = dz_down  # going down; distance to lower level
            eps = - 1E-6


        ######## ADDED May2021 to account for transparent layers: #######
        min_beta = 1E-010
        if mybeta < min_beta: # transparent atmosphere layer
            # update position "for free"
            dztot += dz_tobound + eps  # update deltaz
            new_z += dz_tobound + eps  # update new value of z
            # do not modify photon's energy in this case
        else: ###### END ADDED May2021 ########


            tau_remain = taul - tautemp  # remaining optical depth to integrate
            # print('tau_remain = {}'.format(tau_remain))
            # dlp = tau_remain / (mybeta + 1E-09)  # distance remaining to travel, first trial
            dlp = tau_remain / mybeta  # distance remaining to travel, first trial
            dzp = dlp * cosz  # its vertical component
            # print('dzp = {}'.format(dzp))
            if np.abs(dzp) < np.abs(dz_tobound):  # path ends within this cell
                tautemp += mybeta * np.abs(dlp)  # update path optical thickness
                converged = True  # we are done
                dztot += dzp  # update z-coordinate change
                new_z += dzp  # update new z-coord value
                # energy_kept_fraction = np.exp( - dlp * mybeta_abs)

            else:
                tautemp += mybeta * np.abs(dz_tobound / cosz)  # update opt. thick.
                dztot += dz_tobound + eps  # update deltaz
                new_z += dz_tobound + eps  # update new value of z
                # energy_kept_fraction = np.exp(-dz_tobound/cosz*mybeta_abs)
                # photon.energy = photon.energy * np.exp( - dz_tobound / cosz * mybeta_abs)

            # if photon.frac_energy:
            #     # print('Energy lost frac =', energy_lost_fraction)
            #     photon.Eatm = photon.Eatm + photon.energy * (1 - energy_kept_fraction)
            #     photon.energy = photon.energy * (energy_kept_fraction)

        # if verbose:
        #     print('travelling:: dztot = {}'.format(dztot))
        #     print('travelling:: new_z = {}'.format(new_z))



        # if verbose and new_z < np.min(zz) or new_z > np.max(zz):
        #     print('travelling:: leaving the domain!!!')

        if new_z < np.min(zz):
            impact = True  # leaving the domain from below

        if new_z > np.max(zz):
            escaped = True

    # if verbose:
    # print('dztot = ', dztot)
    # print('new_z = ', new_z)

    dl = dztot / cosz  # compute total path length

    return dl, new_z, impact, escaped, taul, tautemp








@jit(nopython=True)
def upperbounds(hMAX, hMIN, x, y, cosz0, phi0, tilted = True):
    # given a domain x-y at level hMIN
    # compute the corresponding domain at higher level hMAX
    # shifted based on angles cosz9 and phi0.
    if hMAX >= hMIN and tilted == True:
        # dl = np.abs((hMAX - hMIN) / cosz0)
        # sintheta = np.sin(np.arccos(cosz0))  # We are going backward::
        # dx = - dl * sintheta * np.sin(phi0)  # shift in opposite direction compared to travel dir.
        # dy = - dl * sintheta * np.cos(phi0)  # shift in opposite direction compared to travel dir.
        # dl = np.abs((hMAX - hMIN) / cosz0)
        dz = hMAX - hMIN
        # sintheta = np.sin(np.arccos(cosz0))  # We are going backward::
        # dx = - dl * sintheta * np.sin(phi0)  # shift in opposite direction compared to travel dir.
        # dy = - dl * sintheta * np.cos(phi0)  # shift in opposite direction compared to travel dir.
        # theta0 = np.arccos(cosz0)
        # dy = dz*np.tan(theta0)*np.cos(phi0)
        # dx = dz*np.tan(theta0)*np.sin(phi0)
        dy = dz*np.sqrt(1-cosz0**2)/cosz0*np.cos(phi0)
        dx = dz*np.sqrt(1-cosz0**2)/cosz0*np.sin(phi0)
        xTOP0 = x[0] + dx
        yTOP0 = y[0] + dy
        xTOPL = x[-1] + dx
        yTOPL = y[-1] + dy
    else:
        # if hTOA < hMIN or tilted == False:
        xTOP0 = x[0]
        yTOP0 = y[0]
        xTOPL = x[-1]
        yTOPL = y[-1]
        # vertical domain:
    return xTOP0, xTOPL, yTOP0, yTOPL



################################################################################

################### ADAPTIVE INTERSECTION FINDER ALGORITHM   ###################

################################################################################


# @jit(nopython=True)
# def get_block(i, j, nouterb, x, y, Z):
#     nx = Z.shape[0]
#     ny = Z.shape[1]
#     outersizex = nx // (nouterb + 1)
#     outersizey = ny // (nouterb + 1)
#     if i < nouterb - 1:
#         # xi = x[i * outersizex:(i + 1) * outersizex]
#         xi = x[i * outersizex:(i + 1) * outersizex + 1]
#     else:
#         xi = x[i * outersizex:]
#     if j < nouterb - 1:
#         # yj = y[j * outersizey:(j + 1) * outersizey]
#         yj = y[j * outersizey:(j + 1) * outersizey + 1]
#     else:
#         yj = y[j * outersizey:]
#     if i < nouterb - 1 and j < nouterb - 1:
#         # Zij = Z[i * outersizex:(i + 1) * outersizex,
#         #         j * outersizey:(j + 1) * outersizey]
#         Zij = Z[i * outersizex:(i + 1) * outersizex + 1,
#                 j * outersizey:(j + 1) * outersizey + 1]
#     elif i < nouterb - 1 and j == nouterb - 1:
#         # Zij = Z[i * outersizex:(i + 1) * outersizex, j * outersizey:]
#         Zij = Z[i * outersizex:(i + 1) * outersizex + 1, j * outersizey:]
#     elif i == nouterb - 1 and j < nouterb - 1:
#         Zij = Z[i * outersizex:, j * outersizey:(j + 1) * outersizey + 1]
#     # elif i == nouterb - 1 and j == nouterb - 1:
#     else:
#         Zij = Z[i * outersizex:, j * outersizey:]
#     return xi, yj, Zij

# def test_block_sizes(nx, nb1=10, nb2=6, nb3=6):
#     n1 = nx // (nb1 )
#     n2 = n1 // (nb2 )
#     n3 = n2 // (nb3 )
#     return n1, n2, n3
#
# res1 =  test_block_sizes(1201, nb1=10, nb2=6, nb3=6)
# print(res1)

@jit(nopython=True)
def get_block(i, j, nb, x, y, Z):
    # MODIFIED: The beginning of each block in x and y direction must be
    # an EVEN index number in order to keep the correct triangulation
    # by doing so different blocks will overlap
    # but that's ok for the purpose of finding the intersection
    nx = Z.shape[0]
    ny = Z.shape[1]
    assert(i < nb)
    assert(j < nb)
    assert(nb < nx)
    assert(nb < ny)
    dx = nx // (nb)
    dy = ny // (nb)
    # print('dx size', dx)
    # print('dy size', dy)
    startx = i*dx
    endx = (i + 1)*dx + 1
    starty = j * dy
    endy = (j + 1) * dy + 1

    # ALL blocks must start with even indices to keep a consistent triangulation
    if startx % 2 > 0: # odd number
        startx = startx - 1
    if starty % 2 > 0:  # odd number
        starty = starty - 1
    if i < nb - 1:
        xi = x[startx:endx]
    else:
        xi = x[startx:]
    if j < nb - 1:
        yj = y[starty:endy]
    else:
        yj = y[starty:]
    if i < nb - 1 and j < nb - 1:
        Zij = Z[startx:endx, starty:endy]
    elif i < nb - 1 and j == nb - 1:
        Zij = Z[startx:endx, starty:]
    elif i == nb - 1 and j < nb - 1:
        Zij = Z[startx:, starty:endy]
    elif i == nb-1 and j == nb - 1:
        Zij = Z[startx:, starty:]
    else:
        raise Exception("Error!")
    return xi, yj, Zij


@jit(nopython=True)
def distance_point_segment(Q, P1, P2):
    """
    :param Q: Point
    :param P1: Initial Point
    :param P2: Final point of the segment
    :return: distance
    """
    u = P2 - P1
    v = Q - P1
    c1 = mydot(u, v)/mydot(u, u)
    # orthogonal projection of Q on the line::
    P = P1 + c1*u
    if c1 > 0.0 and c1 < 1.0:
        dist = np.sqrt( mydot(P-Q, P-Q))
    # elif c1 < 0.0:
    #     dist = np.sqrt( mydot(P1-Q, P1-Q))
    # elif c1 > 0.0:
    #     dist = np.sqrt( mydot(P2-Q, P2-Q))
    else:
        dist1 = np.sqrt( mydot(P1-Q, P1-Q))
        dist2 = np.sqrt( mydot(P2-Q, P2-Q))
        dist = min(dist1, dist2)
        # dist = np.nan
        # raise Exception('Something has gone terribly wrong! o.0 ')
    return dist




@jit(nopython=True)
def is_block_near_enough(S0, S1, xi, yj, Zij):
    '''
    Returns True if the maximum distance between segment S0-S1 and the surface
    with coordinates xi, yj and elevations Zij exceeds the diagonal of the
    smallest rectangular box containing the surface xi-yj-Zij
    '''
    maxz = np.max(Zij)
    minz = np.min(Zij)
    dv = maxz - minz # no need of abs because get squared
    dbx = xi[-1] - xi[0]
    dby = yj[-1] - yj[0]
    max_distance_allowed = np.sqrt(dbx ** 2 + dby**2 + dv ** 2)
    # print('max dist allowed =', max_distance_allowed)
    # if at least a corner has a larger distance, skip entire block::
    myC = np.array( [xi[0], yj[0], minz] )
    distp = distance_point_segment(myC, S0, S1)
    if distp > max_distance_allowed:
        return False

    # CORNX = np.array([xi[0], xi[-1]])
    # CORNY = np.array([yj[0], yj[-1]])
    # CORNZ = np.array([np.min(Zij), np.max(Zij)])
    # for ci in CORNX:
    #     for cj in CORNY:
    #         for ck in CORNZ:
    #             myC = np.array([ci, cj, ck])
    #             # distp = np.linalg.norm(np.cross(S1-S0,
    #             #       S0-myC))/np.linalg.norm(S1-S0)
    #             distp = distance_point_segment(myC, S0, S1)
    #             # print('myC', myC, 'S0', S0, 'S1', S1, 'distp = ', distp)
    #             if distp > max_distance_allowed:
    #                 return False
    return True


# note: nb1, nb2, nb3 must be carefully chosen based on the tile size
# try beforehand to make sure the combination works and
# it performs well (domain size-depedendent only)

@jit(nopython=True)
def adaptive_intersection_2(S0, S1, x, y, Z, nb1 = 10, nb2 = 6):
    RES=[]
    NORM = []
    for i1 in range(nb1):
        for j1 in range(nb1):
            xi1, yj1, Zij1 = get_block(i1, j1, nb1, x, y, Z)
            # print(Zij1.shape)
            if is_block_near_enough(S0, S1, xi1, yj1, Zij1):
                for i2 in range(nb2):
                    for j2 in range(nb2):
                        xi2, yj2, Zij2 = get_block(i2, j2, nb2, xi1, yj1, Zij1)
                        # Yj2, Xi2 = meshgrid_numba(yj2, xi2)
                        # check intersection with all triangles within the block
                        # save intersection points for all
                        # IS_INT, MYNORM, INTPOINT = check_surface_intersections(
                        #     S0, S1, xi2, yj2, Xi2, Yj2, Zij2,
                        #     usebox=False, check_2d=False)
                        IS_INT, MYNORM, INTPOINT = check_surface_intersections(
                            S0, S1, xi2, yj2, Zij2,
                            usebox=False, check_2d=False)
                        if IS_INT == True:
                            RES.append(INTPOINT)
                            NORM.append(MYNORM)
    # print('-------------------------------------')
    # print('list of int points - 2')
    # print(RES)
    # print('-------------------------------------')
    lenp = len(RES)
    res_IS_INT = True if lenp > 0 else False
    if  res_IS_INT:
        # DISTANCES = [S0.distance(other) for other in RES]
        DISTANCES = np.zeros(lenp)
        for elemi in range(lenp):
            DISTANCES[elemi] = point_distance(S0, RES[elemi])
        mindist_indx = np.int(np.argmin(DISTANCES))
        res_INTPOINT = RES[mindist_indx]
        res_MYNORM = NORM[mindist_indx]
    else:
        res_MYNORM = np.array([-9999.9, -9999.9, -9999.9])
        res_INTPOINT = np.array([-9999.9, -9999.9, -9999.9])
    return res_IS_INT, res_MYNORM, res_INTPOINT


@jit(nopython=True)
def adaptive_intersection_3(S0, S1, x, y, Z, nb1 = 10, nb2 = 6, nb3 = 6):
    #
    RES=[]
    NORM = []
    # results = {}
    for i1 in range(nb1):
        for j1 in range(nb1):
            xi1, yj1, Zij1 = get_block(i1, j1, nb1, x, y, Z)
            # print(Zij1.shape)
            if is_block_near_enough(S0, S1, xi1, yj1, Zij1):
                for i2 in range(nb2):
                    for j2 in range(nb2):
                        xi2, yj2, Zij2 = get_block(i2, j2, nb2, xi1, yj1, Zij1)
                        # print('yj2', Zij2.shape)
                        if is_block_near_enough(S0, S1, xi2, yj2, Zij2):
                            for i3 in range(nb3):
                                for j3 in range(nb3):
                                    xi3, yj3, Zij3 = get_block(i3, j3, nb3,
                                                               xi2, yj2, Zij2)
                                    # print('shape Z3 =', np.shape(Zij3))
                                    # print('yj3', Zij3.shape)
                                    if is_block_near_enough(S0, S1,
                                                            xi3, yj3, Zij3):
                                        # Yj3, Xi3 = meshgrid_numba(yj3, xi3)
                                        # check intersection with all triangles
                                        # save intersection points for all
                                        # IS_INT, MYNORM, INTPOINT = check_surface_intersections(
                                        #     S0, S1, xi3, yj3, Xi3, Yj3, Zij3,
                                        #     usebox=False, check_2d=False)
                                        IS_INT, MYNORM, INTPOINT = check_surface_intersections(
                                            S0, S1, xi3, yj3, Zij3,
                                            usebox=False, check_2d=False)
                                        if IS_INT == True:
                                            RES.append(INTPOINT)
                                            NORM.append(MYNORM)
    # results['is_intersected'] = True if len(RES) > 0 else False
    # if results['is_intersected']:
    #     # DISTANCES = [S0.distance(other) for other in RES]
    #     DISTANCES = np.zeros(len(RES))
    #     for elemi in range(len(RES)):
    #         DISTANCES[elemi] = point_distance(S0, RES[elemi])
    #     mindist_indx = np.int(np.argmin(DISTANCES))
    #     results['int_point'] = RES[mindist_indx]
    #     results['MYNORM'] = NORM[mindist_indx]
    # return results

    # print('-------------------------------------')
    # print('list of int points - 3')
    # print(RES)
    # print('-------------------------------------')
    lenp = len(RES)
    res_IS_INT = True if lenp > 0 else False
    if  res_IS_INT:
        # DISTANCES = [S0.distance(other) for other in RES]
        DISTANCES = np.zeros(lenp)
        for elemi in range(lenp):
            DISTANCES[elemi] = point_distance(S0, RES[elemi])
        mindist_indx = np.int(np.argmin(DISTANCES))
        res_INTPOINT = RES[mindist_indx]
        res_MYNORM = NORM[mindist_indx]
    else: # no intersection found
        res_MYNORM = np.array([-9999.9, -9999.9, -9999.9])
        res_INTPOINT = np.array([-9999.9, -9999.9, -9999.9])
    return res_IS_INT, res_MYNORM, res_INTPOINT


@jit(nopython=True)
def meshgrid_numba(x, y):
    nx = np.shape(x)[0]
    ny = np.shape(y)[0]
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    for i in range(ny):
        X[i, :] = x
    for j in range(nx):
        Y[:, j] = y
    return X, Y

# X2, Y2 = meshgrid_numba(x, y)

# if res['is_intersected']:
#     self.is_impacted = True
#     self.last_impact = res['int_point']
#     self.last_normal = res['MYNORM']



@jit(nopython=True)
def get_atm_value(z, param='wc', dfatm=None):
    # supported values => 'wc', 'tt', 'pr', 'zz'.
    # get the value of an atmospheric parameter
    # at the specified elevation level [m over msl]
    # return the first value for the layer with lower bound
    # closest below the given z-value

    if param in ['zz', 'pr']: # layers not levels,
        # in this case skip first value i.e. return the lower value
        vals = dfatm[param][1:]
    else:
        vals = dfatm[param]

    if z >= np.max( dfatm['zz'] ):
        value = vals[0] # second (lower) value for level values
    elif z <= np.min( dfatm['zz'] ):
        value = vals[-1]
    else:
        # indx = np.where( z < np.array(dfatm['zz']))[0][-1]
        indx = np.where( z < dfatm['zz'])[0][-1]
        value = vals[indx]

        # added as an option: Interpolate
        # not used since the values provided are within each atm. layer
        # interpolate = True
        # if interpolate:
        #     indx_lower = indx + 1
        #     indx_upper = indx
        #     value_lower = vals[indx_lower]
        #     value_upper = vals[indx_upper]
        #
        #     z_upper = dfatm['zz'][indx_upper]
        #     z_lower = dfatm['zz'][indx_lower]
        #     dz_lower = z - z_lower
        #     dz_upper = z_upper - z
        #     assert dz_upper > 0
        #     assert dz_lower > 0
        #     dz_tot = dz_upper + dz_lower
        #     value = value_lower * (dz_lower/dz_tot) + value_upper * (dz_upper/dz_tot)
        # else:
        #     value = vals[indx]

    return value
