#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=100GB

#filename
echo "$0"
printf "%s" "$(<$0)"
echo ""

module load python
#DEBUG: Run to test local analysis
#srun -n 192 -c 2 python receiver_brute_mpi.py --debug

#NORMAL: Run to test streaming with adios2 analysis
srun -n 1 python generator_brute.py
