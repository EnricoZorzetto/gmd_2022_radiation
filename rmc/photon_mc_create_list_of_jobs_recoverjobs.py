import os
import sys
import re
import json
import pandas as pd
import numpy as np
from itertools import product

from shutil import rmtree
# import time


def init_simulation_wrapper(metadata):
    # PASS IN INPUT THE METADATA DICT FROM THE JSON FILE
    # read parameters of the simulation array
    njobs_each_case = metadata['njobs_each_case']
    ADIR = np.array(metadata['ADIR'])
    PHI = np.array(metadata['PHI_OVER_2PI'])*2*np.pi
    COSZ = np.array(metadata['COSZ'])
    BANDS = np.array(metadata['BANDS'])
    datadir = metadata['datadir']
    exp_name = metadata['name']
    # directory to save the results to
    outdir = os.path.join(datadir, 'output_{}'.format(exp_name))
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # if os.path.exists(outdir):
    #     rmtree(outdir, ignore_errors=True) # don't complain!
    # os.makedirs(outdir)

    njobs_3D, df_3D = init_simulation_jobs(COSZ,
                                           PHI = PHI,
                                           ADIR = ADIR,
                                           BANDS = BANDS,
                                           njobs_each_case=njobs_each_case,
                                           planepar = False
                                           )

    njobs_PP, df_PP = init_simulation_jobs(COSZ,
                                           PHI = np.array([0]),
                                           ADIR = ADIR,
                                           BANDS = BANDS,
                                           njobs_each_case=1,
                                           planepar=True
                                           )

    # create single dataset for all jobs - rename index to identify outputs
    planepar = metadata['planepar']
    if len(planepar) == 2:
        df = pd.concat( [df_3D, df_PP], ignore_index=True)
        numjobs = njobs_PP + njobs_3D
    elif len(planepar)==1 and planepar[0] == 0:
        df = df_3D
        numjobs = njobs_3D
    elif len(planepar)==1 and planepar[0] == 1:
        df = df_PP
        numjobs = njobs_PP
    else:
        raise Exception('Invalid value provided for input "planepar"!')


    df['JOBID'] = df.index
    outfilename = os.path.join(outdir, 'numberofjobs.csv')
    fl = open( outfilename, "w")
    fl.write("{}".format(numjobs))
    fl.close()
    df.to_csv( os.path.join(outdir, 'list_jobs_params_original.csv'), index=False)

    # to re-do failed jobs if simulation goes overtime:
    # add here a second csv with the JOBID without results in output_temp
    runjobs = [re.split('_|.',fname)[3] for fname in list(os.listdir(outdir))]
    df_torun = df[df['JOBID'] not in runjobs]
    numjobs_torun = df_torun.shape[0]
    print("numjobs to re-run = ", numjobs_torun)
    df_torun.to_csv( os.path.join(outdir, 'list_jobs_params.csv'), index=False)


    ############################################################################
    print(outdir) ######## NECESSARY TO PASS IT TO THE BASH SCRIPT #############
    ############################################################################
    return numjobs_torun

def init_simulation_jobs(COSZ = None ,
                         PHI = None,
                         ADIR = None,
                         BANDS = None,
                         planepar = True,
                         njobs_each_case = 2):
    '''-------------------------------------------------------------------------
    construct a dataframe df with the parameters to pass to each job
    (only those that differ job-to-job:
    e.g., sun angles (COSZ, PHI), direct albedo (ADIR).
    For now set the diffuse albedo later based on the ADIR value
    By default, COSZ and PHI as in Lee et al., 2011
    # add wavelength information in the future
    -------------------------------------------------------------------------'''
    # print('Initializing the paramters for the jobs')
    nalbedos = len(ADIR)
    nazimuth = len(PHI)
    nzenith = len(COSZ)
    nbands = len(BANDS)
    NJOBSC = np.arange(njobs_each_case)
    ncases = nalbedos*nazimuth*nzenith
    njobs = njobs_each_case*ncases*nbands
    DF = []
    for ib in range(nbands):
        prod_list_0 = list(product(COSZ, PHI, ADIR, NJOBSC))
        df = pd.DataFrame(prod_list_0, columns=['cosz', 'phi', 'adir', 'njobc'])
        # df['myband'] = np.repeat(BANDS[ib], df.shape[0])
        df['myfreq'] = np.int(BANDS[ib])
        df['cases'] = np.repeat(np.arange(ncases), njobs_each_case)
        # df['cases'] = np.repeat(np.arange(njobs_each_case), ncases)
        df['planepar'] = np.int(planepar)
        DF.append(df)
    # rdf = pd.concat(DF).reset_index()
    rdf = pd.concat(DF, ignore_index=True)
    return njobs, rdf



if __name__ == '__main__':

    if len(sys.argv) < 2:  # no experiment file provided, use the default
        mdfile = os.path.join('exp', 'cluster.json')
    else:  # run as::$ python program.py exp/experiment.json
        # in the case of the cluster, one argument is passed with the json file
        mdfile = sys.argv[1]
    metadata = json.load(open(mdfile, 'r'))

    numjobs = init_simulation_wrapper(metadata)

    # copy the experiment json file in the output folder
    # mdfile_out = os.path.join(outdir, "experiment.json")
    # with open(mdfile, "r") as fromf:
    #     with open(mdfile_out, "w") as tof:
    #         tof.write(fromf.read())
