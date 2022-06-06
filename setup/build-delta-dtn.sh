#!/bin/bash

#here we'll attempt to build a version of delta on the dtns
#that is compatible with the version in our current delta/4:0 container
#current version of build is https://github.com/rkube/delta/blob/master/container/delta-outside/Dockerfile

#since cori scratch is currently visible on the dtns, we'll build there
#since it's faster than GPFS

#we'll assume you want to build from scratch, will remove any 
#existing builds, so be careful

#location of your delta clone
###
### change me!!!
export delta_install=/global/cscratch1/sd/stephey/delta
###
###

export build=$delta_install/build
rm -rf $build
mkdir -p $build

##############mpich#################

cd $build
export mpich=3.3
export mpich_prefix=mpich-$mpich
wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz
tar xvzf $mpich_prefix.tar.gz
cd $mpich_prefix
./configure  --prefix=$build/mpich
make -j 8
make install

export LD_LIBRARY_PATH=$build/mpich/lib:$LD_LIBRARY_PATH
export PATH=$build/mpich/bin:$PATH

#----------------------------------

##############cmake################

#next cmake. maybe could have gotten away with doing it via
#conda but too late

cd $build
wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2.tar.gz
tar xvzf cmake-3.23.2.tar.gz
cd cmake-3.23.2
./bootstrap --prefix=$build/cmake && make -j 8 && make install

export PATH=$build/cmake/bin:$PATH

#----------------------------------

###############zeromq##############

cd $build
wget https://github.com/zeromq/libzmq/releases/download/v4.3.4/zeromq-4.3.4.tar.gz
tar vxf zeromq-4.3.4.tar.gz
cd zeromq-4.3.4
./configure --prefix=$build/zeromq &&  make -j 8 && make install

export LD_LIBRARY_PATH=$build/zeromq/lib:$LD_LIBRARY_PATH

#---------------------------------

##############python##############

module load python
conda create -n delta-dtn python=3.8 -y
source activate delta-dtn
python -m pip install --no-cache-dir -r $delta_install/requirements.txt
#we can get away with this here since we're bringing our own mpich stack
#and actually we don't want to link to cray mpich anyway
python -m pip install mpi4py
export conda_env=$CONDA_PREFIX
conda deactivate
module unload python

#--------------------------------

#############adios2##############

cd $build
wget https://github.com/ornladios/ADIOS2/archive/94c2e377eba2947ae9739c1dd42f616a1296f12a.tar.gz
tar xvf 94c2e377eba2947ae9739c1dd42f616a1296f12a.tar.gz
mv ADIOS2-94c2e377eba2947ae9739c1dd42f616a1296f12a adios2-devel
rm -rf adios2-build
mkdir adios2-build
cd adios2-build
cmake                                                        \
        -DCMAKE_INSTALL_PREFIX=$build/adios2             \
        -DZeroMQ_ROOT=$build/zeromq                      \
        -DZeroMQ_INCLUDE_DIRS=$build/zeromq/include      \
        -DZeroMQ_LIBRARY_DIRS=$build/zeromq/lib          \
        -DPython_ROOT=$conda_env                             \
        -DCMAKE_BUILD_TYPE=Release                           \
        -DBUILD_SHARED_LIBS=ON                               \
        -DADIOS2_USE_MPI=ON                                  \
        -DADIOS2_USE_Python=ON                               \
        -DADIOS2_BUILD_EXAMPLES=OFF                          \
        -DADIOS2_BUILD_TESTING=OFF                           \
        -DADIOS2_USE_ADIOS1=OFF                              \
        -DADIOS2_USE_HDF5=OFF                                \
        -DADIOS2_USE_SST=ON                                  \
        -DADIOS2_USE_Profiling=OFF                           \
        -DADIOS2_USE_DATAMAN=ON                              \
        -DADIOS2_USE_ZeroMQ=ON                               \
        ../adios2-devel
make -j 8 && make install

#--------------------------------
