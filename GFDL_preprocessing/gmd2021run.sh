#!/bin/bash


source /lustre/f2/dev/Enrico.Zorzetto/software/miniconda3/etc/profile.d/conda.sh
conda activate HMC2021mpi


echo "cleaning up old log files..."
rm -f test_gmd2021.err
rm -f test_gmd2021.out
echo "create list of experiment files..."
python experiments/experiments_gmd_2021.py &
pid1 = $!
wait $pid1
sleep 3
echo "running experiments..."
sbatch gmd2021sbatch.sh &
pid2 = $!
wait $pid2
sleep3
echo "extract variables to export and zip the result..."
source light.sh
