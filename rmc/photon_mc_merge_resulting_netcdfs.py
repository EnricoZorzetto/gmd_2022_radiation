import os
import sys
import json
import time
import pandas as pd
import numpy as np
from shutil import rmtree
import xarray as xr
from itertools import product

# AFTER THE MONTE CARLO SIMULATION:
# READ ALL THE RESULTING NETCDF AND MERGE THOSE WITH SAME PARAMETERS

def merge_metcdfs(metadata=None):

    datadir = metadata['datadir']
    exp_name = metadata['name']
    sum_gpoints = metadata['sum_gpoints']
    planepar = metadata['planepar']
    njobs_each_case = metadata['njobs_each_case'] # for 3D simulations only
    outdir = os.path.join(datadir, 'output_{}'.format(exp_name))
    outdir_temp = os.path.join(datadir, outdir, 'output_temp')
    outdir_sim = os.path.join(datadir, outdir, 'output_sim')

    with open( os.path.join(outdir, "experiment.json"), "w") as exp_outfile:
        json.dump(metadata, exp_outfile)

    if not os.path.exists(outdir_sim):
        os.makedirs(outdir_sim)

    if 0 in planepar:
        outdir_sim_3D = os.path.join(outdir_sim, 'output_sim_3D')
        if not os.path.exists(outdir_sim_3D):
            os.makedirs(outdir_sim_3D)

    if 1 in planepar:
        outdir_sim_PP = os.path.join(outdir_sim, 'output_sim_PP')
        if not os.path.exists(outdir_sim_PP):
            os.makedirs(outdir_sim_PP)

    # read list of jobs::
    df = pd.read_csv( os.path.join(outdir, 'list_jobs_params.csv'))

    # planepar_dict = {'3D':df3D, 'PP':dfPP}
    planepar_dict_0 = {'3D':0, 'PP':1}
    planepar_dict = { key:item for (key, item) in planepar_dict_0.items()
                      if item in planepar}

    # read separately 3D and PP results
    for ip in list(planepar_dict.keys()):

        #>> Get only PP or 3D list of jobs::
        mydf = df[df['planepar']==planepar_dict[ip]].copy()

        if sum_gpoints == True: # (i.e., combine results across wavelengths)
            unique_gpoints = np.unique(df['myfreq']) # get list of unique gpoints
            dfres = mydf[(mydf['njobc'] == 0) &
                         (mydf['myfreq']==unique_gpoints[0])
                         ].drop(['njobc','planepar','JOBID','myfreq'], axis=1
                                ).copy()

        else: # (i.e., save results separately for each wavelength)
            mydf['cases'] = pd.factorize(mydf['cases'].astype(str) +
                                         mydf['myfreq'].astype(str))[0]
            dfres = mydf[(mydf['njobc'] == 0)].drop(
                         ['njobc','planepar','JOBID'], axis=1)
  
        dfres.to_csv( os.path.join(outdir,
                'list_sim_cases_{}.csv'.format(ip)), index=False)
        cases = dfres['cases'].values

        for case_ic in cases:
            dfi = mydf[mydf['cases'] == case_ic].copy() # simulations for current case
            dfi.reset_index(inplace=True)
            nfiles_ic =  dfi.shape[0] # number of file/simulations for current case
            # print("nfiles_ic = {}, ip = {}".format(nfiles_ic, ip))
            # copy the first file in output file

            file_ic0 = os.path.join(outdir_temp,
                    'photonmc_output_temp_{}.nc'.format(dfi['JOBID'].iloc[0]))
            file_res = os.path.join(outdir_sim, 'output_sim_{}'.format(ip),
                    'photonmc_output_{}.nc'.format(case_ic)) # ok
            # INITIALIZE OUTPUT SIM DATASET  
            ds0 = xr.open_dataset(file_ic0)
            nx = ds0.dims['lon']
            ny = ds0.dims['lat']
            ECOUP = np.zeros((nx, ny), dtype=np.float32)
            EDIR =  np.zeros((nx, ny), dtype=np.float32)
            EDIF =  np.zeros((nx, ny), dtype=np.float32)
            ERDIR = np.zeros((nx, ny), dtype=np.float32)
            ERDIF = np.zeros((nx, ny), dtype=np.float32)
            etoa = 0
            eabs = 0
            esrf = 0
            nphotons = 0
            ftoanorm_sum = 0

            ds_sim = xr.Dataset(
                data_vars=dict(
                    ecoup=(["lon", "lat"], ECOUP),
                    erdif=(["lon", "lat"], ERDIF),
                    erdir=(["lon", "lat"], ERDIR),
                    edir=( ["lon", "lat"], EDIR),
                    edif=( ["lon", "lat"], EDIF),
                    elev=( ["lon", "lat"], ds0['elev'].values)
                ),
                coords=dict(
                    lat=(["lat"], ds0.coords['lat'].values),
                    lon=(["lon"], ds0.coords['lon'].values),
                    lat_meters=(["lat_meters"], ds0.coords['lat_meters'].values),
                    lon_meters=(["lon_meters"], ds0.coords['lon_meters'].values),
                ),
                attrs=dict(description="Simulation parameters.",
                           # hTOA=ds0.attrs['hTOA'],
                           # hMIN=ds0.attrs['hMIN'],
                           ftoanorm=0,
                           cosz=ds0.attrs['cosz'],
                           phi=ds0.attrs['phi'],
                           alpha_dir=ds0.attrs['alpha_dir'],
                           alpha_dif=ds0.attrs['alpha_dif'],
                           ave_elev=ds0.attrs['ave_elev'],
                           # zelevmax=ds0.attrs['zelevmax'],
                           tilted=ds0.attrs['tilted'],
                           forcepscatter=ds0.attrs['forcepscatter'],
                           pscatterf=ds0.attrs['pscatterf'],
                           aerosol=ds0.attrs['aerosol'],
                           nphotons=0,
                           planepar=ds0.attrs['planepar'],
                           etoa=0,
                           eabs=0,
                           esrf=0
                           ),
            )


            for row in range(nfiles_ic): # the first has already been done

                file_ici = os.path.join(outdir_temp,
                    'photonmc_output_temp_{}.nc'.format(dfi['JOBID'].iloc[row]))

                ds_temp = xr.open_dataset(file_ici)

                # for each case and each band, we just need to sum energy deposition
                FTOANORM = ds_temp.attrs['ftoanorm']
                NPHOTONS = ds_temp.attrs['nphotons'] # for a single simulation file

                print("nfiles_ic = {}, ip = {}, row = {}, FTOANORM = {}".format(nfiles_ic, ip, row, FTOANORM))

                if ip == 'PP': 
                    effective_njobs_each_case = 1
                # elif ip == '3D':
                else:
                    effective_njobs_each_case = njobs_each_case

                # Normlize energy received by the surface based on the
                # flux at the top of atmosphere, not the number of photons
                # print('NJOBS PER CASE::',ip, nfiles_ic)
                # print(njobs_each_case, nfiles_ic)
                # assert njobs_each_case == nfiles_ic # holds only for 3D runs
                normalized_to_ftoanorm = True
                # if "normalized_to_ftoanorm" in metadata.keys():
                #     normalized_to_ftoanorm = metadata["normalized_to_ftoanorm"]

                if normalized_to_ftoanorm:
                    ftoanorm_sum = ftoanorm_sum + ds_temp.attrs['ftoanorm'] / effective_njobs_each_case
                    etoa = etoa + ds_temp.attrs['etoa'] * FTOANORM/NPHOTONS / effective_njobs_each_case
                    esrf = esrf + ds_temp.attrs['esrf'] * FTOANORM/NPHOTONS / effective_njobs_each_case
                    eabs = eabs + ds_temp.attrs['eabs'] * FTOANORM/NPHOTONS / effective_njobs_each_case
                    nphotons = nphotons + ds_temp.attrs['nphotons']
                    ECOUP = ECOUP + ds_temp['ecoup'].values  * FTOANORM/NPHOTONS / effective_njobs_each_case
                    ERDIR = ERDIR + ds_temp['erdir'].values  * FTOANORM/NPHOTONS / effective_njobs_each_case
                    ERDIF = ERDIF + ds_temp['erdif'].values  * FTOANORM/NPHOTONS / effective_njobs_each_case
                    EDIR = EDIR   + ds_temp['edir'].values   * FTOANORM/NPHOTONS / effective_njobs_each_case
                    EDIF = EDIF   + ds_temp['edif'].values   * FTOANORM/NPHOTONS / effective_njobs_each_case
                else:
                    ftoanorm_sum = ftoanorm_sum + ds_temp.attrs['ftoanorm']
                    etoa = etoa + ds_temp.attrs['etoa']
                    esrf = esrf + ds_temp.attrs['esrf']
                    eabs = eabs + ds_temp.attrs['eabs']
                    nphotons = nphotons + ds_temp.attrs['nphotons']
                    ECOUP = ECOUP + ds_temp['ecoup'].values
                    ERDIR = ERDIR + ds_temp['erdir'].values
                    ERDIF = ERDIF + ds_temp['erdif'].values
                    EDIR = EDIR   + ds_temp['edir'].values
                    EDIF = EDIF   + ds_temp['edif'].values

                # check here all variables that should not differ from 2
                # simulations when we are mergine their results
                assert np.isclose(ds_sim.attrs['cosz'], ds_temp.attrs['cosz'])
                assert np.isclose(ds_sim.attrs['phi'], ds_temp.attrs['phi'])
                assert np.isclose(ds_sim.attrs['alpha_dir'], ds_temp.attrs['alpha_dir'])
                assert np.isclose(ds_sim.attrs['alpha_dif'], ds_temp.attrs['alpha_dif'])

                # assert np.isclose(ds_sim.attrs['hTOA'], ds_temp.attrs['hTOA'])
                # assert np.isclose(ds_sim.attrs['hMIN'], ds_temp.attrs['hMIN'])
                # assert np.isclose(ds_sim.attrs['zelevmax'], ds_temp.attrs['zelevmax'])
                assert np.isclose(ds_sim.attrs['tilted'], ds_temp.attrs['tilted'])
                assert np.isclose(ds_sim.attrs['planepar'], ds_temp.attrs['planepar'])
                assert np.isclose(ds_sim.attrs['forcepscatter'], ds_temp.attrs['forcepscatter'])
                assert np.isclose(ds_sim.attrs['pscatterf'], ds_temp.attrs['pscatterf'])

                # print('phi:: ds_sim = {}, ds_temp = {}'.format(
                #     ds_sim.attrs['phi'],
                #     ds_temp.attrs['phi'],
                # ))
                # print('nphotons = {}'.format(ds_sim.attrs['nphotons']))
                # print('sum dir = {}'.format(np.sum(ds_sim['edir'].values)))

                ds_temp.close()

            # moved earlier
            # if ip == '3D': # for 'PP' this is always 1 in practice
            #     ftoanorm_sum = ftoanorm_sum / njobs_each_case
            # elif ip == 'PP': # for 'PP' this is always 1 in practice
            #     assert njobs_each_case == 1

            # print("ip = {}, ftoanorm_sum = {}, nphotons = {}".format(ip, ftoanorm_sum, nphotons))

            ds_sim.attrs['ftoanorm'] = ftoanorm_sum
            ds_sim.attrs['etoa'] = etoa
            ds_sim.attrs['eabs'] = eabs
            ds_sim.attrs['esrf'] = esrf
            ds_sim.attrs['nphotons'] = nphotons
            ds_sim['ecoup'].values = ECOUP
            ds_sim['erdif'].values = ERDIF
            ds_sim['erdir'].values = ERDIR
            ds_sim['edir'].values  = EDIR
            ds_sim['edif'].values  = EDIF
            # print('eTOA::', ds_sim.attrs['etoa'])
            # print('nphotons::', ds_sim.attrs['nphotons'])
            # print('edir::', np.sum(ds_sim['edir'].values))
            ds_sim.to_netcdf(file_res)
            ds_sim.close()

    # NOW CLEAN REMOVE ALL TEMPORARY FILES

    time.sleep(5) # sleep for 5 sdeconds
    rmtree(outdir_temp, ignore_errors=True) # don't complain!



if __name__ == '__main__':
    # to run from cluster
    mdfile = sys.argv[1]
    metadata = json.load(open(mdfile, 'r'))
    merge_metcdfs(metadata=metadata)
