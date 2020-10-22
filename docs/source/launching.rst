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
    module unload PrgEnv-cray PrgEnv-gnu PrgEnv-intel
    module load PrgEnv-gnu
    module unload craype-hugepages2M 
    module unload python
    module load python3
    module use -a /global/cscratch1/sd/jyc/sw/modulefiles
    module load adios2/devel
    module load python_delta_comm

    export OMP_NUM_CORES=4
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores 

    srun -n 16 -c 4 --cpu-bind=cores python -m mpi4py.futures processor.py --config configs/delta_config.json


2-node scenario
###############
In this scenario, the `generator` streams data directly to the `proceessor`.

The recommended way is to start the `processor` on cori. You may need to increase the
timeout in the message queues to accomodate a longer wait for initial time chunks.
After the processor has started, launch the generator on the KSTAR DTN:

The commands below set up the environment on the KSTAR DTN to start the generator

.. code-block:: shell

    module use -a /home/choij/sw/spack/share/spack/modules/linux-centos7-haswell
    module use -a /home/choij/sw/spack/share/spack/modules/linux-centos7-broadwell
    module use -a /home/choij/sw/modulefiles

    module load openmpi
    module load zeromq adios2
    module load python py-numpy py-mpi4py py-h5py py-scipy

    mpirun -n 1 python generator.py --config configs/test_2node.json




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
    module load zeromq
    module load python py-numpy py-mpi4py py-h5py py-scipy py-matplotlib py-pyyaml
    module use -a /global/cscratch1/sd/jyc/dtn/sw/modulefiles
    module load adios2
    module load python_delta_comm

    python middleman.poy --config configs/test_3node.json
