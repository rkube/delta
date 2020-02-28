# DELTA-FUSION (aDaptive rEaL Time Analysis of big fusion data)

This project implements a client-server model for analysis of streaming data from
fusion experiments or large-scale simulations.

Implemented as part of "Adaptive near-real time net-worked analysis of big
fusion data", (FY18).

This project implements a streaming analysis workflow. Data is streamed by a generator, using the new
ADIOS2 WAN capabilities, to a processing facility. At the processing facility, the received data packets
are analyzed and stored by a backend. Optional visualization can be attached by coupling to the backend.

This repository is a loose collection of 
* generators
* processors
* backends

The implemented analysis routines are based on [https://www.github.com:minjunJchoi/fluctana](https://www.github.com:minjunJchoi/fluctana) refactored in cython and adapted as computational kernels

Generators, processors, and backends read their configuration from a shared json file. The different
implementations don't have a common syntax yet.

(https://github.com/rkube/delta/blob/master/Streaming%20analysis%20architecture.png)

# Implemented Workflows

## Workflow Scenario #1 (2-node scenario)
In this scenario, data is streamed from a KSTAR Data Transfer Node (DTN) to a NERSC DTN:

```
  generator.py         =====>    processor_xyz.py
(running on KSTAR DTN)   |     (running on NERSC DTN)
                         v                      
     stream name: shotnum-channelid.bp          
```
Processors implement distributed computing in different ways xyz=[mpi, mpi_brute, dask, ...]
Here mpi refers to the new mpi implementation with cython kernels, mpi_brute refers to the
brute-force adaption of the fluctana routines by wrapping them in mpi and dask refers to 
an implementation using dask-distributed.
As of 2020-02, the dask-distributed implementation is out-dated.



Here is an example configuration file for the generator running on the KSTAR DTN:
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

### Jongs reference implementation of the 2-node workflow
This reference implmentation shows the feasability of streaming the data from KSTAR to NERSC with
high velocity. To run this scenaris log in to the respective DTNS and execute:

```
python generator.py --config config-jychoi.json
python receiver.py --config config-jychoi.json
```

Note that the processor is called receiver

### MPI processor 
This is the xyz=mpi case. 
processor_mpi.py implements this case. This processor
* Reads ECEI data from a bp file
* Puts the data in a queue
* A worker thread reads data from the queue and dispatches it to a MPI Executor for analysis
* Calls the multi-threaded cython kernels for data processing

Run this implemntation as
```
export OMP_NUM_THREAD=N
srun -n 6 -c N python -m mpi4py.futures processor_mpi.py --config configs/test_crossphase.json
```

For the KNL nodes, best performance is with N=8/16 and 24 or 48 MPI ranks.

Data storage is implmented for numpy and mongodb backends. See the configuration files configs/test_all.json.


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

# Workflow Scenario #2 (3-node scenario)
This scenario adds an additional station, from the NERSC DTNs and the Cori compute nodes.
Data streamed to the DTN and then forwarded to the processor running on the compute nodes.
This mitigates the low bandwidth available to the compute nodes to the outside.


```
  generator.py         =====>    receiver.py         =====>  analysis.py
(running on KSTAR DTN)   |     (running on NERSC DTN)  |      (running on NERSC compute nodes)
                         v                             v
     stream name: shotnum-channelid.bp          shotnum-channelid.s1.bp
```
This scenario is currently not fully.
