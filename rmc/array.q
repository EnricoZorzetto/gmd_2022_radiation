#!/bin/bash
#SBATCH -c 1
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
eval "$(conda shell.bash hook)"
conda activate rad
echo "using the following metadata:"
echo $1
python main_photon_cluster.py $1

