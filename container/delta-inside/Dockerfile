#needs to be devel since cupy requires libcublas which shifter doesn't currently provide
FROM docker.io/nvidia/cuda:11.0-devel-ubuntu20.04

WORKDIR /opt

#ubuntu 20.04 bugfix
ENV DEBIAN_FRONTEND noninteractive

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        cmake                \
        gcc                  \
        gfortran             \
        python3-dev          \
        python3-pip          \
        libzmq3-dev          \
        wget              && \
    apt-get clean all

#install mpich
ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 32                                                              && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

WORKDIR /opt

#install adios from current master
RUN wget https://github.com/ornladios/ADIOS2/archive/master.tar.gz    && \
    tar xvzf master.tar.gz                                            && \
    mv ADIOS2-master adios2-devel                                     && \
    #python3 -m pip install --no-cache-dir -r requirements.txt         && \
    mkdir adios2-build                                                && \
    rm -rf master.tar.gz

#delta must come before adios2

#first update pip
RUN python3 -m pip install -U setuptools pip

#note this is not cuda-aware
RUN python3 -m pip install mpi4py

#here we actually install delta and its requirements
#note cuda is 11.0 so cupy is 110
RUN wget https://github.com/rkube/delta/archive/master.tar.gz          && \
    tar xvzf master.tar.gz                                             && \
    mv delta-master delta                                              && \
    rm -rf master.tar.gz                                               && \
    cd delta                                                           && \
    python3 -m pip install --no-cache-dir -r requirements.txt          && \
    cd delta/analysis                                                  && \
    CC=cc LDSHARED="cc -shared" python3 setup.py build_ext --inplace


WORKDIR /opt/adios2-build

ENV VER docker

ENV PREFIX /software/adios2-$VER

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
        -DADIOS2_USE_ZeroMQ=ON                               \
        ../adios2-devel

RUN make -j 32; make install

RUN ln -s /software/adios2-$VER /software/adios2-current

#this looks different in ubuntu
ENV PYTHONPATH /software/adios2-current/lib/python3/dist-packages

WORKDIR /opt

#mongodb python driver
RUN wget https://github.com/mongodb/mongo-python-driver/archive/master.tar.gz   && \
    tar xvzf master.tar.gz                                                      && \
    mv mongo-python-driver-master mongo-python-driver                           && \
    rm -rf master.tar.gz                                                        && \
    cd mongo-python-driver                                                      && \
    > bson/_cbsonmodule.c                                                       && \
    > bson/buffer.c                                                             && \
    > bson/buffer.h                                                             && \
    > pymongo/_cmessagemodule.c                                                 && \
    python3 setup.py build                                                       && \
    python3 setup.py install --user

