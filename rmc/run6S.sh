# Running a photomc simulation

eval "$(conda shell.bash hook)"
conda activate rad


# clean up result files if already existing
rm -rf slurmout
echo "cleaning up directory slurmout"
mkdir slurmout
#

# modify here to change experiment: e.g., exp/pp.json, exp/3d.json
#EXPFILE="exp/3d.json" # ADD YOUR EXPERIMENT FILE HERE
#EXPFILE="exp/cluster_gpoints.json" # ADD YOUR EXPERIMENT FILE HERE
EXPFILE="exp/cluster_test_py6S.json" # ADD YOUR EXPERIMENT FILE HERE
echo "using the following metadata:"
echo $EXPFILE
#OUTDIR=$(python create_list_of_jobs.py $EXPFILE)
OUTDIR=$(python photon_mc_create_list_of_jobs.py $EXPFILE)


wait


#rm -r $OUTDIR
#echo "created output directory $OUTDIR"

#mkdir -p $OUTDIR
TEMPFOLDER="${OUTDIR}/output_temp"
#rm -r $TEMPFOLDER
mkdir -p $TEMPFOLDER # not done in python 'cause it is within the array!

SIMFOLDER="${OUTDIR}/output_sim"
rm -rf $SIMFOLDER
#mkdir $SIMFOLDER # done in the python code!
#echo "creating folder ${OUTDIR}"
echo "creating folder ${TEMPFOLDER}"

#nj=$(cat "../../dem_datasets/output_3D/numberofjobs.csv")
nj=$(cat "${OUTDIR}/numberofjobs.csv")
echo "Number of jobs to execute is $nj"
njm1=$(($nj-1))

echo $njm1
echo $EXPFILE

# launch array of jobs - max use 128 cores at the same time
#sbatch --array 0-$njm1 --parsable array2.q
# REMEMBER THAT IN NATE's CLUSTER THE LIMIT SIZE OF JOB ARRAYS IS 1000
JOBID=$(sbatch -o slurmout/array_%A_%a.out -e slurmout/array_%A_%a.err --array 0-$njm1 --parsable array.q "$EXPFILE")
#JOBID=$(sbatch -o slurmout/array_%A_%a.out -e slurmout/array_%A_%a.err --array 0-$njm1%20 --parsable array.q "$EXPFILE")
#sbatch -o out -e err --array 0-10 --parsable array.q exp/cluster.json
#JOBID=$(sbatch -o array.out -e array.err --array 0-$njm1 --parsable array.q "$EXPFILE")
#JOBID=$(sbatch --array 0-$njm1 array.q "$EXPFILE")

echo "running job array number $JOBID"

#Wait for entire job array to complete successfully

sbatch --depend=afterok:$JOBID combine.sh "$EXPFILE"

wait


date