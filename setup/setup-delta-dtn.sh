#!/bin/bash

#location of your delta clone
###
### change me!!!
export delta_install=/global/cscratch1/sd/stephey/delta
###
###
export build=$delta_install/build

#source custom python env
module load python
source activate delta-dtn

#point to adios2 python lib
export PYTHONPATH=$build/adios2/lib/python3.8/site-packages/adios2:$PYTHONPATH

#for debugging:
#export LD_LIBRARY_PATH=$build/mpich/lib:$LD_LIBRARY_PATH
#export PATH=$build/mpich/bin:$PATH
