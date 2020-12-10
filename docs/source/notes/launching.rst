*********************************************
Launching data analysis workflows using Delta
*********************************************




1-node scenario
###############
In this scenario Delta will run on a single node only. This is the recommended way
to test data analysis workflows.

The commands below set up the environment on cori to start the processor:

.. code-block:: shell

    #!/bin/bash
    export OMP_NUM_CORES=16
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores 

    srun -n 4 -c 16 --cpu-bind=cores python -m mpi4py.futures processor.py --config configs/delta_config.json


2-node scenario
###############
In this scenario, the `generator` streams data directly to the `proceessor`.

The recommended way is to start the `processor` on Cori. You may need to increase the
timeout in the message queues to accomodate a longer wait for initial time chunks.
After the processor has started, launch the generator on the sending system:

The commands below set up the environment on the KSTAR DTN to start the generator

.. code-block:: shell

    module use -a /home/choij/sw/spack/share/spack/modules/linux-centos7-broadwell

    module load openmpi
    module load zeromq 
    module load py-numpy py-mpi4py py-h5py py-scipy py-matplotlib py-pyyaml py-more-itertools

    module use -a /home/choij/sw/modulefiles
    module load adios2

    mpirun -np 1 python generator.py --config configs/test_all.json --kstar


.. note::

    To use SST with Cori, we need to do as follows:
    Use ``--sdn``  option with ``salloc``  (which is to get an IP address to be visible from the DTN):

    .. code-block:: shell

       salloc --sdn ...

    Then, run srun  as follows:

    .. code-block:: shell

        ADIOS2_IP=$SDN_IP_ADDR srun ...

    IPAddress in the config is for DataMan and it should be the ip address of the sender (i.e., the dtn nodes)


.. note::

    IPAddress in the config is for DataMan and it should be the ip address of the sender. 
    Between NERSC DTN and Cori, it should be the ip addresss of one of NERSC DTN nodes. To use between KSTAR, it should be the ip address of the KSTAR DTN node (see below).




3-node scenario
###############
In this scenario the `generator` streams data to a `middleman` which forwards the
packages to the `processor`.

The recommended way is to start the `processor` first, then the `middleman`, and
finally the `generator`.

The commands below set up the environment on the NERSC DTN to start the middleman

.. code-block:: shell

    #!/bin/bash
    module use -a /global/cscratch1/sd/jyc/dtn/sw/spack/share/spack/modules/linux-centos7-ivybridge

    module load openmpi
    module load libzmq
    module load python py-numpy py-mpi4py py-h5py py-scipy py-matplotlib py-pyyaml

    module use -a /global/cscratch1/sd/jyc/dtn/sw/modulefiles
    module load adios2
    module load python_delta_comm

    mpirun -np 1 python middleman.py --config configs/test_all.json 
