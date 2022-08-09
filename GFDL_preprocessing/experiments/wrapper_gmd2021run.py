import os
import sys
# from mpi4py import MPI

# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))
# print("Hello world!")

# # the first sys argument is always the name of the file:
# script_name = sys.argv[0]
# print("running script {}".format(script_name))

# IF running this jobs as follows: mpirun -n 4 run_mpi "python test.py"
# then the 2nd argument is the job rank, and the 3rd the job size
# if inclusing other sys args, they go first
numjob = int(sys.argv[1])
batchsize = int(sys.argv[2])
# print('Index for current JOB MPI = {} out of {}'.format(numjob, batchsize))

# print('other 0', sys.argv[0])
# print('other 5', sys.argv[5])

# comm = MPI.COMM_WORLD
# myrank = comm.Get_size()
# myrank = comm.Get_rank()
# print('job rank = {}'.format( myrank))

list_oj_jobs_file = open('/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/gmd_2021_grids/list_of_exp_files.txt')

expfiles_all = list_oj_jobs_file .readlines()
expfile = expfiles_all[numjob]
print(numjob, expfile)
# to do the production run::
os.system("python driver.py {}".format(expfile))