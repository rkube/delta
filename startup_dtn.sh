#!/bin/bash
module use -a /global/cscratch1/sd/jyc/dtn/sw/spack/share/spack/modules/linux-centos7-ivybridge
module load openmpi
module load zeromq
module load python py-numpy py-mpi4py py-h5py py-scipy py-matplotlib py-pyyaml
module use -a /global/cscratch1/sd/jyc/dtn/sw/modulefiles
module load adios2
module load python_delta_comm
