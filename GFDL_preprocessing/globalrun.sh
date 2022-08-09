#!/bin/bash
#SBATCH --job-name=HMC_GFDL
#SBATCH --error=hmc_global.err
#SBATCH --output=hmc_global.out
#SBATCH --clusters=c3
#SBATCH --account=cpo_3dland
#SBATCH --nodes=48
#SBATCH --ntasks=1536
#SBATCH --ntasks-per-node=32
#SBATCH --qos=windfall
#SBATCH --time=0-05:00:00

# outdir='/stor/soteria/hydro/shared/enrico/res_hmc/'
outdir='/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/'

mkdir -p $outdir
echo "creating output folder ${outdir} ..."
# clean old result folder and create a new folder

source /lustre/f2/dev/Enrico.Zorzetto/software/miniconda3/etc/profile.d/conda.sh
# conda activate ezdev
conda activate HMC2021mpi


#srun python /home/ez23/HMC_GFDL/driver.py "exp_gmd/res_${exper}.json"
# srun python /lustre/f2/dev/Enrico.Zorzetto/Projects/GFDL_preprocessing/driver.py "experiments/globe1.json"
python /lustre/f2/dev/Enrico.Zorzetto/Projects/GFDL_preprocessing/driver.py "experiments/globe1.json"

#srun echo "I shall process the input file" $exper




