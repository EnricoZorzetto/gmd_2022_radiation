#!/bin/bash
#SBATCH --clusters=c3
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=3
#SBATCH --error=slurmout/hmc_mpi2.err
#SBATCH --output=slurmout/hmc_mpi2.out
#SBATCH --account=cpo_3dland
#SBATCH --qos=windfall
#SBATCH --time=0-00:19:00
########SBATCH --time=1-00:00:00
########SBATCH --qos=windfall

# echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
eval "$(conda shell.bash hook)"
conda activate rmc5
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

