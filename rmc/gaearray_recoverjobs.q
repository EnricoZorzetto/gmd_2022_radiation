#!/bin/bash
#SBATCH --clusters=c3
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=30
#SBATCH --error=hmc_mpi.err
#SBATCH --output=hmc_mpi.out
#SBATCH --account=cpo_3dland
#SBATCH --time=01:00:00

# echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
eval "$(conda shell.bash hook)"
conda activate radmpi
IFS=';' read -r -a ARGS <<< "$1"
# ARGS=(${$1//;/ })
ARG0=${ARGS[0]}
ARG1=${ARGS[1]}

echo "1"
echo $1
echo "ARGS"
echo $ARGS
echo "using the following metadata file:"
echo $ARG0
echo "number of jobs submitted:"
echo $ARG1
# python main_photon_cluster.py $1
# mpirun -n $ARG1 run_mpi "python $ARG0"
mpirun -n $ARG1 run_mpi "python main_photon_cluster.py $ARG0"

