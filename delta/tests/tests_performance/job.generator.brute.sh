#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=100GB

#filename
echo "$0"
printf "%s" "$(<$0)"
echo ""

module load python
#NORMAL: Run to test streaming with adios2 analysis
srun -n 1 -c 2 python ~/delta_rmchurch/generator_brute.py --config config.generator.brute.json
