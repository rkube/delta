#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --constraint=haswell
#SBATCH --array=0-3

cd /global/homes/r/rkube/repos/delta/rq_tests
conda activate delta
srun /global/homes/r/rkube/.conda/envs/delta/bin/rq worker high default low -u redis://cori02
