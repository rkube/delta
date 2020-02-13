# DELTA-FUSION (aDaptive rEaL Time Analysis of big fusion data)

This project implements a client-server model for analysis of streaming data from
fusion experiments or large-scale simulations.

Implemented as part of "Adaptive near-real time net-worked analysis of big
fusion data", (FY18).


The current implementation features a data generator and a data processor.
Both are MPI applications. Each generator task reads data from a single group of channels
and writes them via the ADIOS2 dataman interface.
Each processor task reads this data and performs a single analysis routine.
The output of the analysis is stored in a database for later analysis and visualization.


To run the analysis framework run a generator and a processor simultaneously on a node:
```
srun -n 2 -c 2 --mem=1G --gres=craynetwork:0 python generator_adios2.py
srun -n 2 -c 2 --mem=1G --gres=craynetwork:0 python processor_adios2.py
```

To have generator and processor side-by-side within an interactive session it is convenient 
to [split the terminal using screen]
(https://unix.stackexchange.com/questions/7453/how-to-split-the-terminal-into-more-than-one-view)


# Workflow Scenario #1 (2-node scenario)
```
  generator.py         =====>    processor_xyz.py
(running on KSTAR DTN)   |     (running on NERSC DTN)
                         v                      
     stream name: shotnum-channelid.bp          
```
xyz=[mpi, mpi_brute, dask, ...]


In this scenario, the processor reads a configuration file, performs the analysis routines
defined in that file, and stores the data for subsequent analysis.


As of now, there is no common format for configuration file. Each processor and receiver has its own
format. 



Parameters can be provided with a json file. Here is an example:
```
{
    "datapath": "/home/choij/kstar_streaming/018431",
    "shotnr": 18431,
    "channel_lists": [[2203, 2204]],
    "analysis": [{"name" : "power_spectrum", 
               "config" : {"nperseg": 32, "fs": 1.0}}],
    "engine": "DataMan", 
    "params": {"IPAddress": "203.230.120.125", 
                "OpenTimeoutSecs": "600"},
    "nstep": 100,
    "analysis_engine": "BP4"

}
```




## Currently implemented processors

### Jongs reference implementation.
Example commands are as follows:
```
python generator.py --config config-jychoi.json
python receiver.py --config config-jychoi.json
```

Note that the processor is called receiver

### MPI processor 
processor_mpi_mockup.py: Implements a naked reference implementation that
* Reads dummy data, mimicking a received data package from KSTAR, in the main loop
* Puts the dummy data in a queue
* A worker thread reads data from the queue and dispatches it to a MPIPoolExecutor for 
  analysis

Run this implemntation as
```
export OMP_NUM_THREAD=1
srun -n 16 python -m mpi4py.futures processor_mpi_mockup.py --config configs/test_crossphase.json
```

The number of OpenMP threads needs to be small since the linked BLAS libraries from
some numpy packages are inherently multithreaded. This causes the processor to run
with a larger number of tasks than available on the machine.


### MPI processor brute
RMC's implementation of fluctana in the framework
```
  generator_brute.py   =====>    receiver_brute.py
(running on KSTAR DTN)   |     (running on NERSC DTN)
                         v
     stream name: shotnum-ch00000.bp
```

We can run as follows. 

First, on a Cori DTN node, run as follows:
```
module use -a /global/cscratch1/sd/jyc/dtn/sw/spack/share/spack/modules/linux-centos7-ivybridge
module use -a /global/cscratch1/sd/jyc/dtn/sw/modulefiles

module load openmpi
module load zeromq adios2
module load python py-numpy py-mpi4py py-h5py py-scipy py-matplotlib

mpirun -n 5 python -u -m mpi4py.futures receiver_brute.py --config config-dtn.json 
```

Then, on KSTAR, run as follows:
```
module use -a /home/choij/sw/spack/share/spack/modules/linux-centos7-haswell
module use -a /home/choij/sw/spack/share/spack/modules/linux-centos7-broadwell
module use -a /home/choij/sw/modulefiles

module load openmpi
module load zeromq adios2
module load python py-numpy py-mpi4py py-h5py py-scipy py-matplotlib

python -u generator_brute.py --config config-kstar.json
```

Here is config files used in the above:
`config-dtn.json`:
```
{
    "datapath": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/",
    "shotnr": 18431,
    "channel_range": ["ECEI_L0101-2408"],
    "analysis": [{"name" : "all"}],
    "fft_params" : {"nfft": 1000, "window": "hann", "overlap": 0.5, "detrend" :1},
    "engine": "DataMan",
    "params": { "IPAddress": "203.230.120.125",
                "Timeout": "60",
                "OneToOneMode": "TRUE",
                "OpenTimeoutSecs": "600"},
    "nstep": 200,
    "batch_size": 10000,
    "resultspath": "./",
}

```
`config-kstar.json`:
```
{
    "datapath": "/home/choij/kstar_streaming/",
    "shotnr": 18431,
    "channel_range": ["ECEI_L0101-2408"],
    "analysis": [{"name" : "all"}],
    "fft_params" : {"nfft": 1000, "window": "hann", "overlap": 0.5, "detrend" :1},
    "engine": "DataMan",
    "params": { "IPAddress": "203.230.120.125",
                "Timeout": "60",
                "OneToOneMode": "TRUE",
                "OpenTimeoutSecs": "600"},
    "nstep": 200,
    "batch_size": 10000,
    "resultspath": "./",
}
```

# (obsolete) Workflow Scenario #2 (3-node scenario)
It consists of three components:
```
  generator.py         =====>    receiver.py         =====>  analysis.py
(running on KSTAR DTN)   |     (running on NERSC DTN)  |      (running on NERSC compute nodes)
                         v                             v
     stream name: shotnum-channelid.bp          shotnum-channelid.s1.bp
```
This scenario is currently not implemented fully.
