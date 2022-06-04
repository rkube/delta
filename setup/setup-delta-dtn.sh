#!/bin/bash

export build_dir=/global/cscratch1/sd/stephey/delta/build_dir

#source custom python env
module load python
source activate delta-dtn

#point to adios2 python lib
export PYTHONPATH=$build_dir/adios2/lib/python3.8/site-packages/adios2:$PYTHONPATH
