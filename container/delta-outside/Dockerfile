FROM nvcr.io/nvidia/cuda:11.5.0-devel-ubuntu20.04

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
        git                  \
        python3-dev          \
        python3-pip          \
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
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

ARG mpi4py=3.1.3
ARG mpi4py_prefix=mpi4py-$mpi4py

RUN /sbin/ldconfig

#install mpi4py manually, may not be necessary but let's be careful for now
RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    python3 setup.py build                                                    && \
    python3 setup.py install                                                  && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix                                                    && \
    cd $HOME

#install delta requirements using file in local repo
#this makes it possible to adjust requirements that have
#not been merged upstream yet
RUN python3 -m pip install -U setuptools pip
ADD requirements.txt /opt/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

#KSTAR DTN has 4.3.2, it's very important to match!
ARG zmq_version=4.3.2
RUN wget https://github.com/zeromq/libzmq/releases/download/v$zmq_version/zeromq-$zmq_version.tar.gz && \
    tar vxf zeromq-$zmq_version.tar.gz && \
    cd zeromq-$zmq_version && \
    ./configure --prefix=/opt/zeromq &&  make -j 32 && make install
ENV CMAKE_PREFIX_PATH=/opt/zeromq:$CMAKE_PREFIX_PATH

#for the moment i think we're ok without the requirements since
#we install mpi4py ourselves and delta installs numpy
#install adios from 2.7.1 release
RUN git clone https://github.com/ornladios/ADIOS2.git adios2-devel && \    
    cd adios2-devel && \
    git checkout v2.7.1-279-gd65478233 && \
    cd .. && \
    mkdir adios2-build             

RUN mkdir -p /opt/adios2-container

ENV PREFIX /opt/adios2-container

RUN cd /opt/adios2-build                                  && \
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
        -DZeroMQ_INCLUDE_DIR=/opt/zeromq/include             \
        ../adios2-devel                                   && \
        make -j 32                                        && \
        make install      
  

RUN ln -s /opt/adios2-container /opt/adios2-current

RUN /sbin/ldconfig

#path to adios2.so library
ENV PYTHONPATH /opt/adios2-current/lib/python3.8/site-packages/adios2

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

#make the delta dir so we can bind mount it in while running in shifter
RUN mkdir /delta


