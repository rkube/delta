FROM nvidia/cuda:11.0-devel-ubuntu20.04

WORKDIR /tmp

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

#install python packages from requirements file
RUN python -m pip install -U setuptools pip
ADD requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#install adios
RUN wget https://github.com/ornladios/ADIOS2/archive/master.tar.gz    && \
    tar xvzf master.tar.gz                                            && \
    mv ADIOS2-master adios2-devel                                     && \
    mkdir adios2-build                                                && \
    rm -rf master.tar.gz

WORKDIR /tmp/adios2-build

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

RUN make ; make install

RUN ln -s /software/adios2-$VER /software/adios2-current

#this looks different in ubuntu
ENV PYTHONPATH /software/adios2-current/lib/python3/dist-packages

WORKDIR /tmp

#install delta
RUN wget https://github.com/rkube/delta/archive/master.tar.gz          && \
    tar xvzf master.tar.gz                                             && \
    mv delta-master delta                                              && \
    rm -rf master.tar.gz                                               && \
    cd delta                                                           && \
    cd delta/analysis                                                  && \
    CC=cc LDSHARED="cc -shared" python setup.py build_ext --inplace

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
    python setup.py build                                                       && \
    python setup.py install --user

