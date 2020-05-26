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

Here is a diagram of the framework:
![Delta Architecture](https://github.com/rkube/delta/blob/master/doc/delta_architecture.png "Delta Architecture")

# Implemented Workflows

## Flexible workflow (2-node scenario)
In this scenario, data is streamed from a DTN to Cori. The DTN executes generator.py, processor_mpi.py runs on Cori:

```
  generator.py        =======>     processor_mpi.py
    (DTN)                |            (Cori)
                         v
          stream_name: SSSSS_NN.bp
```
The processor accesses distributed computing resources through a thread-pool, see PEP-3148. For Cori,
mpi4py's MPICommExecutor is recommended. But other compatible executors should work as well.
``processor_mpi.py`` implements this flexible workflow. It
* Reads ECEI time chunks from an adios2 source
* Puts the time chunks in a queue
* A worker thread reads time chunks data from the queue, and passes it to Executor for analysis through task_ecei objects.
* Calls the multi-threaded C/cython kernels for data processing

This workflow allows for flexible channel sizes, so we call it the flexible workflow.

Some spectral analysis are implemented as C kernels and interfaced via cython.
To compile the C kernels
```
cd analysis/lib
make
```
To build the cython interface
```
cd analysis
CC=cc LDSHARED="cc -shared" python setup.py build_ext --inplace
```

Run this implementation on cori:
```
module unload PrgEnv-cray PrgEnv-gnu PrgEnv-intel
module load PrgEnv-gnu
module unload craype-hugepages2M
module unload python
module load python3
module use -a /global/cscratch1/sd/jyc/sw/modulefiles
module load adios2/devel
module load python_delta_comm

export OMP_NUM_THREAD=N
srun -n 6 -c N python processor_mpi.py --config configs/test_all.json

Alternatively, run the 2-node scenario using the task_list processor which uses an extra fft executor:
srun -n 6 -c N python -m mpi4py.futures processor_mpi_tasklist.py --config configs/test_all.json
```

## Flexible workflow (3-node scenario)
In this scenario, a middle-man forwards data from the DTN to Cori. This is necessary when the data stream
needs to take another hop:

```
  generator.py   =======>     middle-man.py =====>  processor_mpi.py
    (DTN)           |            (Hop)        |
                    v                         v
          stream_name: SSSSS_NN.bp          stream_name: SSSSS_NN.bp
```

## Setup








# Configuration









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

### Reference implementation of the 2-node workflow
This reference implmentation shows the feasability of streaming the data from KSTAR to NERSC with
high velocity. To run this scenaris log in to the respective DTNS and execute:

```
python generator.py --config config-jychoi.json
python receiver.py --config config-jychoi.json
```

Note that the processor is called receiver


Run the generator on the NERSC DTN as 
```
module use -a /global/cscratch1/sd/jyc/dtn/sw/spack/share/spack/modules/linux-centos7-ivybridge
module load openmpi
module load zeromq
module load python py-numpy py-mpi4py py-h5py py-scipy py-matplotlib py-pyyaml
module use -a /global/cscratch1/sd/jyc/dtn/sw/modulefiles
module load adios2
module load python_delta_comm
```


For the KNL nodes, best performance is with N=8/16 and 24 or 48 MPI ranks.

Data storage is implmented for numpy and mongodb backends. See the configuration files configs/test_all.json.
The mongodb backend allows to store data either internally using gridFS or on the filesystem using numpy.
Using mongodb with the numpy backend still stores all metadata in mongodb. 
Using the pure numpy backend stores also the metadata in numpy files.


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
module load python py-numpy py-mpi4py py-h5py py-pyyaml py-scipy py-matplotlib

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


# Interfacing to visualizers

Visualizers interface to the streaming data analysis through the database. Using the mongodb backend allows
for siphoning the analysis as it comes in through change streams:
[https://docs.mongodb.com/manual/changeStreams/](https://docs.mongodb.com/manual/changeStreams/)

Let's say `processor_mpi.py` is running with runID `1ABCDE`, writing updates into mongodb. By default,
delta writes the analysis results into the collection test_analysis_RUNID, where RUNID is just the 6-char
runID. In a python session you can follow them 
```
cursor = db.test_analysis_1ABCDE.watch()
for change in cursor():
    print(change)
    print("")
```

Visualizers are implemented separately, see f.ex. the web-based dashboard: [https://github.com/rkube/dashboard_v2](https://github.com/rkube/dashboard_v2)
