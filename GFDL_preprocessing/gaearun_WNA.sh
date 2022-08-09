#!/bin/bash
#SBATCH --job-name=HMC_GFDL
#SBATCH --error=hmc4.err
#SBATCH --output=hmc4.out
#SBATCH --clusters=c3
#SBATCH --ntasks=64
#SBATCH --nodes=2
#SBATCH --time=0-01:00:00
#SBATCH --account=cpo_3dland


## remove existing files
#cd /stor/soteria/hydro/shared/enrico/
#
## cleanup old stuff and create new folder for storing results
#rm -rf res_hmc
#
#cd /home/ez23/HMC_GFDL/
# run one by one:
#python driver.py exp_gmd/res_EastAlps_10.json
#python driver.py exp_gmd/res_EastAlps_10.json
#python driver.py exp_gmd/res_EastAlps_10.json
#python driver.py exp_gmd/res_EastAlps_10.json
#python driver.py exp_gmd/res_EastAlps_200.json

#echo "SLURM_ARRAY_TASK_ID::"
#echo $SLURM_ARRAY_TASK_ID
#case $SLURM_ARRAY_TASK_ID in
#   0)  SEED=123 ;;
#   1)  SEED=38  ;;
#   2)  SEED=22  ;;
#   3)  SEED=60  ;;
#   4)  SEED=432 ;;
#esac

#srun python slurm/pi.py 2500000 --seed=$SEED > pi_$SEED.json
#echo $exper

# outdir='/stor/soteria/hydro/shared/enrico/res_hmc/'
outdir='/lustre/f2/dev/Enrico.Zorzetto/preproc_grids/'

mkdir -p $outdir
echo "creating output folder ${outdir} ..."
# clean old result folder and create a new folder

source /lustre/f2/dev/Enrico.Zorzetto/software/miniconda3/etc/profile.d/conda.sh
# conda activate ezdev
conda activate HMC2021mpi


#srun python /home/ez23/HMC_GFDL/driver.py "exp_gmd/res_${exper}.json"
#srun python /lustre/f2/dev/Enrico.Zorzetto/Projects/GFDL_preprocessing/driver.py "experiments/chaney2017_global_dev_mar2021_ezfolder.json"
# python /lustre/f2/dev/Enrico.Zorzetto/Projects/GFDL_preprocessing/driver.py "experiments/point22.json"
python /lustre/f2/dev/Enrico.Zorzetto/Projects/GFDL_preprocessing/driver.py "experiments/cont_ALP.json"
# python /lustre/f2/dev/Enrico.Zorzetto/Projects/GFDL_preprocessing/driver.py "experiments/cont_TPL.json"

#srun echo "I shall process the input file" $exper




