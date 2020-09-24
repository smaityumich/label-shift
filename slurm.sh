#!/bin/bash
#SBATCH --job-name=ls-sim
#SBATCH --output=logs/ls-sim_%A_%a.out
#SBATCH --array=0-3199
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=00:20:00
#SBATCH --account=yuekai1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=smaity@umich.edu
#SBATCH --partition=standard
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

python3 experiment.py $SLURM_ARRAY_TASK_ID
python3 experiment2.py $SLURM_ARRAY_TASK_ID
