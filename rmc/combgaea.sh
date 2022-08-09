#!/bin/bash
#SBATCH --clusters=es
#SBATCH --partition=eslogin
#SBATCH --ntasks=1 
#SBATCH -o slurmout/comb.out
#SBATCH -e slurmout/comb.err
#SBATCH --time=0-00:59:00

eval "$(conda shell.bash hook)"
conda activate rmc5

echo "using the following metadata:"
echo $1
python photon_mc_merge_resulting_netcdfs.py $1
