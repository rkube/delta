FROM nvidia/cuda:11.0-devel-ubuntu20.04
WORKDIR /tmp

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gcc                  \
        gfortran             \
        python3-dev          \
        python3-pip          \
        libzmq3-dev          \
        wget              && \
    apt-get clean all

ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

ARG mpi4py=3.0.0
ARG mpi4py_prefix=mpi4py-$mpi4py

#alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

#install mpi4py manually, may not be necessary but let's be careful for now
RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    python setup.py build                                                    && \
    python setup.py install                                                  && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix                                                    && \
    cd $HOME

RUN /sbin/ldconfig

##install cuda sdk which is (maybe?) required by pip numba
##see here https://numba.pydata.org/numba-doc/dev/user/installing.html#installing-using-pip-on-x86-x86-64-platforms
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run && \
#    /bin/bash cuda_11.2.0_460.27.04_linux.run

#install rest of python packages
RUN python -m pip install -U setuptools pip

RUN pip install             \
        --no-cache-dir      \
        numpy               \
        scipy               \
        more-itertools      \
        pyyaml              \
        scikit-image        \
        cython              \
        h5py                \
        pymongo             \
        tqdm                \
        pytest              \
        pytest-cov          \
        mock                \
        numba               \
        cupy-cuda110

WORKDIR $HOME

# ------------- broken past here

#install adios
RUN wget https://github.com/ornladios/ADIOS2/archive/master.zip adios2-devel    && \
    #need to untar, etc
    cd adios2-devel

ENV VER $(git describe)                                                         && \
    cd ..                                                                       && \
    mkdir adios2-build                                                          && \
    cd adios2-build

ENV PREFIX $HOME/software/adios2-$VER

RUN \
    CC=cc cmake                                              \
        -DCMAKE_INSTALL_PREFIX=$PREFIX                       \
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
        -DZeroMQ_LIBRARY=$HOME/software/zeromq/lib/libzmq.so \
        -DZeroMQ_INCLUDE_DIR=$HOME/software/zeromq/include   \

RUN ../adios2-devel                                                                                && \
    rm $HOME/software/adios2-current                                                               && \
    ln -s $HOME/software/adios2-$VER $HOME/software/adios2-current                                 && \
    export PYTHONPATH=$PYTHONPATH:$HOME/software/adios2-current/lib/python3.9/site-packages/adios2 && \


#now install delta itself
RUN wget https://github.com/rkube/delta/archive/master.zip          && \
    cd delta/analysis                                               && \
    CC=cc LDSHARED="cc -shared" python setup.py build_ext --inplace

#and finally maybe mongodb?
