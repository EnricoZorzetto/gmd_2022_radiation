#!/bin/bash
#SBATCH --job-name=HMC_GFDL
#SBATCH --error=test_gmd2021.err
#SBATCH --output=test_gmd2021.out
#SBATCH --clusters=c3
#SBATCH --account=cpo_3dland
#SBATCH --qos=windfall
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-01:00:00


# source /lustre/f2/dev/Enrico.Zorzetto/software/miniconda3/etc/profile.d/conda.sh
# conda activate ezdev
# conda activate HMC2021
source /lustre/f2/dev/Enrico.Zorzetto/software/miniconda3/etc/profile.d/conda.sh
# conda init
conda activate HMC2021mpi




# srun -n 64 mpi_run "python test.py"
# mpirun -n 64 mpi_run "python test.py"
# mpirun -n 120 run_mpi "experiments/python wrapper_gmd2021run.py"
echo "running the job in parallel for each experiment config ..."
mpirun -n 120 run_mpi "python experiments/wrapper_gmd2021run.py"



