#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-25
#SBATCH -o experiments/logs/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e experiments/logs/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=FAIL

# Activate virtual env so that run_experiment can load the correct packages
source activate dac

experi="1D"

expdir="experiments/other_${experi}-DIST"

cmd="dac/train/train_other.py -r 1 -n 100000 --bo -s ${SLURM_ARRAY_TASK_ID} --env ${experi} --out-dir ${expdir}"

if [ $SLURM_ARRAY_TASK_ID -le 25 ]; then
    python $cmd 
    exit $?
fi
