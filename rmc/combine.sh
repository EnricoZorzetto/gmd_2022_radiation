#!/bin/bash
#SBATCH -c 1
#SBATCH -o slurmout/comb.out
#SBATCH -e slurmout/comb.err
eval "$(conda shell.bash hook)"
conda activate rad

echo "using the following metadata:"
echo $1
python photon_mc_merge_resulting_netcdfs.py $1
