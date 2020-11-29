***********************
Setting up to run Delta
***********************

Prerequisites
#############


It is easiest to run Delta from within in a python virtual environment. To install the dependencies
execute

.. code-block:: shell

    pip install -r requirements.txt

Besides the packets listed in the file `requirements.txt`, `Delta` requires
* mpi4py
* `adios2 <https://adios2.readthedocs.io/en/latest/>`_ with SST and Dataman support enabled.

To compile adios2, this script may be helpful:

.. code-block:: shell

    #!/bin/bash
    [ ! -d adios2-devel ] && git clone https://github.com/ornladios/ADIOS2.git adios2-devel
    cd adios2-devel
    pwd
    VER=$(git describe)
    cd ..

    rm -rf adios2-build
    mkdir adios2-build
    cd adios2-build
    pwd

    PREFIX=$HOME/software/adios2-$VER
    echo $PREFIX
    CC=cc cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_Python=ON \
    -DADIOS2_BUILD_EXAMPLES=OFF \
    -DADIOS2_BUILD_TESTING=OFF \
    -DADIOS2_USE_ADIOS1=OFF \
    -DADIOS2_USE_HDF5=OFF \
    -DADIOS2_USE_SST=ON \
    -DADIOS2_USE_Profiling=OFF \
    -DADIOS2_USE_DATAMAN=ON \
    -DZeroMQ_LIBRARY=$HOME/software/zeromq/lib/libzmq.so \
    -DZeroMQ_INCLUDE_DIR=$HOME/software/zeromq/include \
    ../adios2-devel

    rm $HOME/software/adios2-current
    ln -s $HOME/software/adios2-$VER $HOME/software/adios2-current 

It fetches the most current adios2 version, configures the build with SST and DATAMAN enabled,
and builds it. Please note this DATAMAN requires `ZeroMQ <https://zeromq.org/>`_  

Installing adios2 in this manner requires to set the `PYTHONPATH` environemnt variable. Using the
configuration from the script above this can be done with

.. code-block:: shell

    export PYTHONPATH=$PYTHONPATH:$HOME/software/adios2-current/lib/python3.9/site-packages/adios2



Obtaining Delta
###############

The newest version of Delta can be obtained from github:

.. code-block:: shell

    git clone git@github.com:rkube/delta.git

After fetching the code, the multi-threaded spectral analysis kernels need to be compiled:

.. code-block:: shell

    cd delta/analysis
    CC=cc LDSHARED="cc -shared" python setup.py build_ext --inplace

The kernels are to be compiled with OpenMP. The file `setup.py` includes OpenMP command line
options, hard-coded for GCC. If you plan to use another compiler you may need to change them.


Running pymongo in an MPI environment
#####################################

We observed that pymongo segfaults when executed by mpi4py, see this 
`bug report <https://jira.mongodb.org/projects/PYTHON/issues/PYTHON-2438>`_

Fix: replace all occurances of 'buffer_new' in the pymongo source code with
something that doesn't collide with other libraries, such as 'buffer_new_mongo':


.. code-block:: shell

    git clone https://github.com/mongodb/mongo-python-driver.git
    cd mongo-python-driver/
    module swap PrgEnv-intel PrgEnv-gnu
    module unload craype-hugepages2M
    vi bson/_cbsonmodule.c
    vi bson/buffer.c
    vi bson/buffer.h
    vi pymongo/_cmessagemodule.c
    python setup.py build
    python setup.py install --user

