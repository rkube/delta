#!/bin/bash
conda activate openmpi
export PYTHONPATH=$PYTHONPATH:/global/homes/r/rkube/software/adios2-current/lib/python3.9/site-packages/adios2
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8                                                                                                 
srun -n 8 -c 8 --cpu-bind=cores python -m mpi4py.futures processor.py --config configs/test_all_v2.json
