

# Running a photomc simulation

start_time="$(date -u +%s)"

eval "$(conda shell.bash hook)"
conda activate rmc5


# clean up result files if already existing
rm -rf slurmout
echo "cleaning up directory slurmout"
mkdir slurmout
#

# modify here to change experiment: e.g., exp/pp.json, exp/3d.json
#EXPFILE="exp/3d.json" # ADD YOUR EXPERIMENT FILE HERE
EXPFILE="exp/gaea_prova.json" # ADD YOUR EXPERIMENT FILE HERE
# EXPFILE="exp/gaea_prova.json" # ADD YOUR EXPERIMENT FILE HERE
#EXPFILE="exp/cluster_gpoints.json" # ADD YOUR EXPERIMENT FILE HERE
#EXPFILE="exp/cluster_test_py6S.json" # ADD YOUR EXPERIMENT FILE HERE
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

#NATE
# REMEMBER THAT IN NATE's CLUSTER THE LIMIT SIZE OF JOB ARRAYS IS 1000
# JOBID=$(sbatch -o slurmout/array_%A_%a.out -e slurmout/array_%A_%a.err --array 0-$njm1%120 --parsable array.q "$EXPFILE")


# USING c3 cores on Gaea (32 cpus/node)
# USING c4 cores on Gaea (36 cpus/node)
#JOBID_00=$(sbatch --ntasks-per-node=32 --account=cpo_3dland --nodes=1 --clusters=c3 -o slurmout/array_%A_%a.out -e slurmout/array_%A_%a.err --array 0-$njm1 --parsable array.q "$EXPFILE")
# JOBID_00=$(sbatch --account=cpo_3dland --ntasks-per-node=32 --nodes=2 --clusters=c3 -o slurmout/array_%A_%a.out -e slurmout/array_%A_%a.err --array 0-$njm1 --parsable array.q "$EXPFILE")

# JOBID_00=$(sbatch --parsable gaearray.q "$EXPFILE;$nj")
JOBID_00=$(sbatch --parsable gaearray_prova.q "$EXPFILE;$nj")


# JOBID_00=$(echo $JOBID_0 | tr ";" "\n")
JOBID_0=(${JOBID_00//;/ })
JOBID=${JOBID_0[0]}

# JOBID=$(sbatch -D `pwd` gaea_array.q --export=v1=$EXPFILE,v2=$njm1)
# sbatch gaea_array.q "$EXPFILE" "$njm1"
# sbatch --export=1="$EXPFILE",2="$njm1" gaea_array.q
# JOBID=$(sbatch gaea_array.q "$EXPFILE $njm1")

echo "$JOBID_00"
echo "$JOBID_0"
echo "$JOBID"

echo "running job array number $JOBID"

#Wait for entire job array to complete successfully

sbatch --depend=afterok:$JOBID combgaea.sh "$EXPFILE"

wait

end_time="$(date -u +%s)"

elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"


date
