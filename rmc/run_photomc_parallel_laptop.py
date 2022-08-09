
import os
import sys
import json
import numpy as np
from functools import partial


import photon_mc_create_list_of_jobs as io
# import photon_mc_land as land
# import photon_mc_atmosphere as atm
# import photon_mc_numba_fractional as ph
import photon_mc_merge_resulting_netcdfs as merge
import main_photon_cluster as photomc
import multiprocessing

import time
import xarray as xr
import matplotlib.pyplot as plt

import photon_mc_land as land

"""
Same workflow as done in the cluster (slurm) 
"""


# def target_fun2(iterab):
#     print('hello world {}'.format(iterab))
#     return

def fast_plot(tempdir, job_index, outfigdir):
    # read and plot results:
    init_time = time.time()
    ds = xr.open_dataset(os.path.join(tempdir, 'photonmc_output_{}.nc'.format(job_index)))

    print('****************************************************')
    print('launch time = {} minutes'.format(
        (time.time() - init_time) / 60.0))
    print('****************************************************')

    print(list(ds.variables))
    print(list(ds.attrs))

    y = ds['lat']
    x = ds['lon']
    Z = ds['elev']

    eTOA = ds.attrs['etoa']
    eSRF = ds.attrs['esrf']
    eABS = ds.attrs['eabs']

    ecoup = ds['ecoup'].values
    edir = ds['edir'].values
    edif = ds['edif'].values
    erdir = ds['erdir'].values
    erdif = ds['erdif'].values
    etot = edir + edif + erdir + erdif + ecoup

    print('sum ECOUP = ', np.sum(ecoup))
    print('sum EDIR = ', np.sum(edir))
    print('sum EDIF = ', np.sum(edif))
    print('sum REDIR = ', np.sum(erdir))
    print('sum REDIF = ', np.sum(erdif))
    print('etot = ', np.sum(etot))
    print('****************************************************')
    print('photons escaped = ', eTOA)
    print('photons absorbed = ', eABS)
    print('photons impacted = ', eSRF)

    print("number of hits::")
    print("nedir", np.size(edir[edir>0]))
    print("nedif", np.size(edir[edif>0]))
    print("nerdir", np.size(erdir[erdir>0]))
    print("nerdif", np.size(erdif[erdif>0]))
    print("necoup", np.size(ecoup[ecoup>0]))
    print("nphotons", ds.attrs['nphotons'])
    print("ftoanorm", ds.attrs['ftoanorm'])

    # print('Profiling memory usage: [for linux systems Only]')
    # mem = ph.memory_usage()
    # print(mem)
    # print('****************************************************')

    Y, X = np.meshgrid(y, x)

    # EDIR2 = (edir).astype(float)
    # EDIR2[EDIR2 < 1E-6] = np.nan

    edir2 = edir.copy()
    edir2[edir2 < 1E-6] = np.nan

    plotlast = True
    if plotlast:
        plt.figure()
        # plt.imshow(Z, alpha = 0.6, extent=(x[0], x[-1], y[-1], y[0]))
        # plt.imshow(Z, alpha = 0.6)
        # plt.pcolormesh(y, x, Z, alpha = 0.6, shading='flat')
        # cbar = plt.colorbar()
        # cbar.set_label('Elev. [m msl]')
        # cbar.set_label('Elev. [m msl]')
        plt.ylabel('x EAST [DEM grid points]')
        plt.xlabel('y NORTH [DEM grid points]')
        plt.imshow(edir2, cmap='jet')
        # plt.pcolormesh(y, x, EDIR2[:nby, :nbx])
        # plt.pcolormesh(y, x, edir2)
        # plt.title('cosz = {}, $\phi$ = {}, $\\alpha$ = {}, '
        #           'noscatter'.format(-cosz, phi, alpha_dir))
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(outfigdir, 'impacts.png'), dpi=300)
        plt.show()


        plt.figure()
        plt.pcolormesh(X, Y, Z, shading='auto')
        plt.savefig(os.path.join(outfigdir, 'impacts.png'), dpi=300)
        plt.show()
    return


def main():

    if len(sys.argv) < 2:  # no experiment file provided, use the default
        mdfile = os.path.join('exp', 'laptop.json')
    else:  # run as::$ python program.py exp/experiment.json
        mdfile = sys.argv[1]
    # mdfile = os.path.join('exp', 'laptop.json')
    print('using experiment file = {}'.format(mdfile))



    # print('mdfile = {}'.format(mdfile))
    metadata = json.load(open(mdfile, 'r'))
    numjobs = io.init_simulation_wrapper(metadata)

    exp_name = metadata['name']
    datadir = metadata['datadir']
    outdir = os.path.join(datadir, 'output_{}'.format(exp_name))
    outfigdir = os.path.join(outdir, 'outfigdir')

    tempdir = os.path.join(outdir, 'output_temp')
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    if not os.path.exists(outfigdir):
        os.makedirs(outfigdir)

    # photomc.run_photomc(0, metadata=metadata)
    # exit()

    # NEW: EA DEM, must do preprocessing here
    # land.preprocess_land_dem(metadata)
    land.preprocess_land_dem(metadata)

    ncpus = multiprocessing.cpu_count()
    print('number of cpus available = {}'.format(ncpus))

    iterable = np.arange(numjobs)
    print('iterable = ', iterable)
    pool = multiprocessing.Pool(processes=ncpus)
    tic = time.time()
    target_fun = partial(photomc.run_photomc, metadata=metadata)
    # def target_fun(numjob):
    #     return photomc.run_photomc(numjob, metadata=metadata)
    pool.map(target_fun, iterable)
    # pool.close()
    # pool.join()
    print('All processes have now completed!')
    toc = time.time()
    deltat = toc-tic
    print(deltat)

    # make sure the jobs are finished at this point
    # merge.merge_metcdfs(metadata=metadata)
    #
    # job_index_2plot = 0
    # fast_plot(tempdir, job_index_2plot, outfigdir)


    # copy the experiment json file in the output folder
    # mdfile_out = os.path.join(outdir, "experiment.json")
    # with open(mdfile, "r") as fromf:
    #     with open(mdfile_out, "w") as tof:
    #         tof.write(fromf.read())






if __name__ == '__main__':
    main()
    # print('Finished multiproecssing')
    # print("Now let's merge the resulting netcdf files:")
    # os.system('photon_mc_merge_resulting_netcdfs.py')

    if len(sys.argv) < 2:  # no experiment file provided, use the default
        mdfile = os.path.join('exp', 'laptop.json')
    else:  # run as::$ python program.py exp/experiment.json
        mdfile = sys.argv[1]
    # mdfile = os.path.join('exp', 'laptop.json')
    print('mdfile = {}'.format(mdfile))
    metadata = json.load(open(mdfile, 'r'))
    merge.merge_metcdfs(metadata=metadata)


    plot= True
    if plot:
        tempdir = os.path.join( metadata['datadir'], 'output_{}'.format(metadata['name']),
                                'output_sim', 'output_sim_PP')
                                # 'output_sim', 'output_sim_3D')

        outfigdir = os.path.join( metadata['datadir'], 'output_{}'.format(metadata['name']),
                                'outfigdir')
        job_index_2plot = 0
        fast_plot(tempdir, job_index_2plot, outfigdir)



