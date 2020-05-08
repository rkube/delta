#!/bin/bash

module unload PrgEnv-cray PrgEnv-gnu PrgEnv-intel
module load PrgEnv-gnu
module unload python
module load python3
module use -a /global/cscratch1/sd/jyc/sw/modulefiles
module load adios2/devel
