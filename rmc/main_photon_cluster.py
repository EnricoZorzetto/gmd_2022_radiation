


import os
import sys
import json
import time
import numpy as np
import xarray as xr
import pandas as pd
import photon_mc_land as land
import photon_mc_atmosphere as atm
import photon_mc_numba_fractional as ph
import pickle


def run_photomc(numjob, metadata = None):


    # print('start metadata::')
    # print(metadata)
    # print('end metadata::')



    # hTOA = metadata['hTOA'] # TOP OF ATM HEIGHT [m]
    # hMIN = metadata['hMIN'] # MIN ELEVATION
    # (must bhe below the minimum surface elevation in the DEM)

    domain = metadata['domain']
    adaptive = metadata['adaptive'] # if true, use a faster algorithm
    # for searching intersection
    # you must set the three values below.
    # If DEM to small and too large values it breaks.
    # nb1=8; nb2=4; nb3=4 # for small domain (0.05 x 0.05)
    nb1=metadata['nb1']
    nb2=metadata['nb2']
    nb3=metadata['nb3'] # for larger domains (0.2 x 0.2)
    # nb1=10; nb2=10; nb3=10 # for much larger domains (1 x 1)
    verbose = metadata['verbose'] # print what the photon is doing
    tilted = metadata['tilted'] # use a tilted domain until reaches bounds
    forcepscatter = metadata['forcepscatter'] # if true,
    # force scattering probability given below
    pscatterf = metadata['pscatterf'] # prob of scattering
    # (used only if forcepscatter = True)
    # enter_vertical = True
    # only if tilted domain, decide if photon enters from vert or horiz. bound.
    # planepar = metadata['planepar'] # if true (=1)# , do anal for flat domain
    # (i.e., plane parallel)

    nphotons_pp = metadata['nphotons_pp']
    nphotons_3d = metadata['nphotons_3d']
    # save_res = True # save result in a file.nc
    datadir = metadata['datadir']
    exp_name = metadata['name']
    exp_domain = metadata['domain']
    const_albedo = metadata['const_albedo']
    # adirmat = None
    # adifmat = None



    brdf = metadata['brdf']
    frac_energy = metadata['frac_energy']
    aerosol = metadata['aerosol']
    atmosphere = metadata['atmosphere'] # can be 'GFDL' or 'Py6S'
    ############################################################################

    ############################################################################


    # else:
    # random.seed(10)
    # create folder for output data and figures


    # outfigdir = os.path.join(datadir, 'outfigdir')
    outdir = os.path.join(datadir, 'output_{}'.format(exp_name))
    # simdir = os.path.join(outdir, 'output_sim')
    tempdir = os.path.join(outdir, 'output_temp')

    # read common parameters from file::
    # using_cluster = True
    # if using_cluster:
    dfjobs = pd.read_csv( os.path.join(outdir,
                'list_jobs_params.csv'))
    dfjobs.set_index('JOBID', inplace=True)
    outer_cosz = dfjobs['cosz'].loc[numjob]
    outer_phi = dfjobs['phi'].loc[numjob]
    myfreq = dfjobs['myfreq'].loc[numjob] # ADDDED
    # planepar = np.bool(dfjobs['planepar'].loc[numjob]) # ADDED
    planepar = bool(dfjobs['planepar'].loc[numjob]) # ADDED
    cosz = -outer_cosz
    if outer_phi < 0:
        phi = outer_phi + 2*np.pi
    else:
        phi = outer_phi
    if planepar:
        nphotons = nphotons_pp
    else:
        nphotons = nphotons_3d
    alpha_dir = dfjobs['adir'].loc[numjob]

    # alpha_dif = dfjobs['adif'].loc[numjob]
    # modified: read adif from input as well.
    # Only adir is used as variable to select output
    # alpha_dif = alpha_dir

    # get the corresponding diffuse albedo if it was specified::
    if 'ADIF' in list(metadata.keys()):
        v_alpha_dir = np.array(metadata['ADIR'])
        alpha_index = np.argmin(np.abs(v_alpha_dir - alpha_dir))
        alpha_dif = np.array(metadata['ADIF'])[alpha_index]
    else:
        # IN NOT SPECIFIED, ASSUME THEY ARE EQUAL
        # print('Diffuse albedo was not provided in the input json file:'
        #       'Let us assume it is equal to the direct one!')
        alpha_dif = alpha_dir


    # else:
    #     # raise Exception('Not running in the cluster!')
    #     print('Not running in the cluster!')
    #     cosz = -0.1
    #     outer_cosz = -cosz
    #     phi = 0.0
    #     outer_phi = phi - np.pi
    #     alpha_dir = 0.5
    #     alpha_dif = alpha_dir
    #     planepar = True
    #     mygpoint = 0
    #     nphotons = nphotons_3d
    #     # if not os.path.exists(tempdir):
    #     #     os.makedirs(tempdir)




    #########################  LOAD LAND MODEL - ELEVATION #####################
    use_preprocessed_dem = True
    if not use_preprocessed_dem:
        if domain =='EastAlps':
            x, y, lon, lat, Z = land.read_flip_complete_srtm(
                # datadir = os.path.join(datadir, 'L33_90m'),
                datadir = os.path.join(datadir, 'N46E012'),
                filename='N46E012.hgt')
        elif domain == 'Nepal':
            #  # old domain
            # x, y, lon, lat, Z = land.read_flip_complete_srtm(
            #     datadir = os.path.join(datadir, 'N28E084'),
            #     filename='N28E084.hgt')
            x, y, lon, lat, Z = land.read_flip_complete_srtm(
                datadir = os.path.join(datadir, 'N29E081'),
                filename='N29E081.hgt')
        elif domain == 'Peru':
            x, y, lon, lat, Z = land.read_flip_complete_srtm(
                datadir = os.path.join(datadir, 'S14W073'),
                filename='S14W073.hgt')
        else:
            raise Exception('Specify a valid domain!')

    else:
        ddir_dem_file = os.path.join(outdir, 'preprocessed_dems',
                                            '{}_dem_input.pkl'.format(domain))
        with open(ddir_dem_file, 'rb') as pklfile:
            demdict = pickle.load(pklfile)
        x = demdict['x']; y = demdict['y']; Z = demdict['Z']
        lon = demdict['lons']; lat = demdict['lats']


    # # to subset the domain: (# TODO: move this to the land module)
    # npixelskept = 100
    # x = x[:npixelskept].copy()
    # y = y[:npixelskept].copy()
    # Z = Z[:npixelskept, :npixelskept].copy()
    # #
    # x, y, Z = land.complete_periodic_edges(
    #     x, y, Z, offsetx=2, offsety=2, plot=False)

    Zmean = np.int32(np.round(np.mean(Z)))
    # print("Zmean = ", Zmean)
    # print('******************************************')
    # Zmean = 10
    # print('WARNING main_photon_cluster.py::::')
    # print('WARNING: USE THE REAL Zmean!!!!!')
    # print('******************************************')
    if planepar:
        Z = np.ones(np.shape(Z), dtype=np.int32)*Zmean

    # print("Zmean = ", Zmean)


    #######################  LOAD LAND MODEL - ALBEDO ##########################
    if const_albedo:
        adirmat = np.zeros((2,2))
        adifmat = np.zeros((2,2))
        xalb = np.arange(2)
        yalb = np.arange(2)
    else:
        yalb, xalb, adirmat, adifmat = land.read_modis_albedos(
            type='winter', plot=False, datadir = datadir)
        xalb = (xalb - xalb[0]) / (xalb[-1] - xalb[0]) * (x[-1] - x[0]) + x[0]
        yalb = (yalb - yalb[0]) / (yalb[-1] - yalb[0]) * (y[-1] - y[0]) + y[0]
        # if const_albedo:
        #     shape_adirmat = adirmat.shape
        #     adirmat = alpha_dir * np.ones(shape_adirmat)
        #     adifmat = alpha_dif * np.ones(shape_adirmat)

    # print('running simulation for inner cosz = {}; '
    #       'inner phi = {}'.format(cosz, phi))
    # print('running simulation for outer cosz = {}; '
    #       'outer phi = {}'.format(outer_cosz, outer_phi))


    ########################## LOAD ATMOSPHERE MODEL ###########################
    # TODO: Would be nice to move this in the preprocessing and here read dfatm
    # TODO: for the specific band (myfreq) needed here
    # my_outer_band = '12850-16000'
    my_outer_band = '16000_22650'
    # add metadata option to chose which atmosphere to use
    if atmosphere == 'GFDL':
        # if metadata['aerosol']:
        #     raise Exception('ERROR: Aerosol not yet available for this '
        #                     'atmospheric profile!')
        dfatm = atm.init_atmosphere_gfdl(metadata,
                                         myband=my_outer_band,
                                         mygpoint=myfreq)

    elif atmosphere == 'GFDL_singletimestep':
        dfatm = atm.init_atmosphere_gfdl_singletimestep(metadata,
                                         myband=my_outer_band,
                                         mygpoint=myfreq)

    elif atmosphere == 'Py6S':
        # if Py6S, must provide the lambda values -> see wavelengths avail in 6S
        # mylambda = 0.550

        ALL_LAMBDAS = np.array([0.350, 0.400, 0.412, 0.443, 0.470, 0.488, 0.515,
                                0.550, 0.590, 0.633, 0.670, 0.694, 0.760, 0.860,
                                1.240, 1.536, 1.650, 1.950, 2.250, 3.750])
        mylambda = ALL_LAMBDAS[myfreq]
        dfatm = atm.init_atmosphere_py6S(mylambda=mylambda,
                                         cosz = outer_cosz, aerosol=aerosol)
    else:
        raise Exception("Error: Must provide a valid atmosphere!")
    # dfatm = atm.init_atmosphere_McClatchey()

    # save the atmospheric profile in the output
    # in multi processes there should be no conflicts (only numjobs=0 writes it)
    # print("dfatm")
    # print(dfatm)
    if numjob == 0:
        dfatm2 = dict(dfatm)
        with open( os.path.join(outdir,
                'dfatm_{}.pickle'.format(numjob)), 'wb') as handle:
            pickle.dump(dfatm2, handle)
    #     dfatm2 = pd.DataFrame(dfatm)
    #     dfatm2.to_csv( os.path.join(outdir, 'dfatm.csv') )

    ############################################################################



    ########################## Running the simulation ##########################
    init_time = time.time()
    # print("*****************")
    # print('aerosol = {}'.format(aerosol))
    # print("*****************")
    # print('start simulate arguments')
    # print(np.shape(x), np.shape(y), np.shape(Z))
    # print(nphotons, cosz, phi, alpha_dir, alpha_dif, yalb, xalb)
    # print(np.shape(adifmat), np.shape(adirmat), brdf, const_albedo)
    # print(frac_energy, aerosol, adaptive, tilted, verbose, pscatterf)
    # print(forcepscatter, nb1, nb2, nb3, np.shape(dfatm))
    # print('end simulate arguments')

    # print(nphotons)
    # print(x[0], y[0], Z[0,0])
    # print(x.shape, y.shape, Z.shape)
    # print(alpha_dir, alpha_dif, cosz, phi)
    # print(yalb, xalb)
    # print(adirmat, adifmat, brdf)
    # print(const_albedo, frac_energy, aerosol)
    # print(adaptive, tilted, verbose)
    # print(pscatterf, forcepscatter, nb1, nb2, nb3)
    # print(dfatm.keys())
    # print(dfatm['zz'].shape)


    myresult = ph.simulate(
        nphotons=nphotons,
        x=x, y=y, Z=Z, cosz=cosz, phi=phi,
        alpha_dir=alpha_dir, alpha_dif=alpha_dif,
        yalb=yalb, xalb=xalb,
        adirmat=adirmat, adifmat=adifmat, brdf=brdf,
        const_albedo=const_albedo, frac_energy=frac_energy,
        aerosol=aerosol,
        adaptive=adaptive, tilted=tilted, verbose=verbose,
        pscatterf=pscatterf, forcepscatter=forcepscatter,
        nb1=nb1, nb2=nb2, nb3=nb3, dfatm=dfatm)

    print('job = {}, nphotons = {}, cosz = {}, simulation time = {} minutes'.format(
        numjob, nphotons, cosz, (time.time() - init_time) / 60.0))


    # print('FTOANORM =', dfatm['rsdcsaf'][0])


    ######################## EXPORT OUTPUT AS NETCDF FILE ######################

    # print('-----------------------------------')
    # print('TOA rsdcsaf:::')
    # print(dfatm['rsdcsaf'][0])
    # print('-----------------------------------')
    ds = xr.Dataset(
        data_vars=dict(
            ecoup=(["lon", "lat"], myresult.ECOUP),
            erdif=(["lon", "lat"], myresult.ERDIF),
            erdir=(["lon", "lat"], myresult.ERDIR),
             edir=(["lon", "lat"], myresult.EDIR),
             edif=(["lon", "lat"], myresult.EDIF),
           elev = (["lon", "lat"], Z)
        ),
        coords=dict(
            lat_meters=(["lat_meters"], y),
            lon_meters=(["lon_meters"], x),
            lat=(["lat"], lat),
            lon=(["lon"], lon),
        ),
        attrs=dict(description="Simulation parameters",
                   # PARAMETERS OF THE SIMULATIONS::
                   # hTOA = hTOA, # top of atm
                   # hMIN = hMIN, # lower level atmosphere
                   ftoanorm=dfatm['rsdcsaf'][0],  # FLUX AT THE TOA
                   freq=myfreq,  # top of atm
                   cosz= outer_cosz,
                   phi = outer_phi,
                   ave_elev = Zmean,
                   alpha_dir = alpha_dir,
                   alpha_dif = alpha_dif,
                   tilted = int(tilted), # use tilted domain
                   forcepscatter = int(forcepscatter),
                   pscatterf = pscatterf,
                   frac_energy = frac_energy,
                   const_albedo = const_albedo,
                   aerosol = aerosol,
                   nphotons=nphotons, # total number launched in current job
                   planepar=int(planepar), # total number in current job
                   # AND SOME SUMMARY RESULTS::
                   etoa=myresult.eTOA, # number of photons escaped from TOA
                   eabs=myresult.eABS, # zerosnumber of photons abs. in atm
                   esrf=myresult.eSRF # total num of photons absorbed by surface
                   ),
    )
    ds.to_netcdf(os.path.join(tempdir,
            'photonmc_output_temp_{}.nc'.format(numjob)))



if __name__ == '__main__':

    # run this only from cluster now / in array / with argument
    # Modified for GAEA
    try:
        # print('running:: main_photon_cluster.py:')
        # print('Try using slurm array first')
        numjob = int(os.environ['SLURM_ARRAY_TASK_ID']) # FOR USING SLURM ARRAYS (DUKE CLUSTER)
    except:   
        # print('running:: main_photon_cluster.py:')
        # print('Tried slurm array without success. Using MPI instead!')
        numjob = int(sys.argv[2]) # INDEX FOR CURRENT JOB IF USING MPI
    # print('This is process number: numjob==', numjob)
    mdfile = sys.argv[1] # EXPERIMENT FILES - DOES NOT CHANGE, ALWAYS ARG POS #1
    metadata = json.load(open(mdfile, 'r'))
    run_photomc(numjob, metadata=metadata)

    # Determine whether code is running on the cluster::
    # but now this will be only called by the cluster
    # try:
    #     numjob = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #     using_cluster = True
    # except KeyError:
    #     using_cluster = False
    #     numjob = np.nan
    # print('using cluster = {}; running job '
    #       'number {}'.format(using_cluster, numjob))
    #
    # if using_cluster:
    #     mdfile = sys.argv[1]
    #     metadata = json.load(open(mdfile,'r'))



    # ############################################################################
    # # READ SIMULATION PARAMETERS FROM THE EXPERIMENT JSON FILE
    # # IF NO FILE GIVEN AS ARGUMENT, USE THE DEFAULT laptop.json for my laptop
    # ############################################################################
    # if not using_cluster:
    #     mdfile = os.path.join('exp', 'laptop.json')
    # else:
    #     if len(sys.argv) < 2: # no experiment file provided, use the default
    #         mdfile = os.path.join('exp', 'laptop.json')
    #     else: # run as::$ python program.py exp/experiment.json
    #         mdfile = sys.argv[1]
    # metadata = json.load(open(mdfile,'r'))
    # # os.system('mkdir -p %s' % metadata['dir'])
    #
    # # os.system('mkdir -p logs')

