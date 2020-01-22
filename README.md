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


# Workflow Scenario #1
This is a three-node scenario, where data is streamed from a KSTAR DTN to a NERSC DTN and
subsequently passed to a compute node:

```
  generator.py         =====>    receiver.py         =====>  analysis_adios2.py (not yet implemented)
(running on KSTAR DTN)   |     (running on NERSC DTN)  |      (running on NERSC compute nodes)
                         v                             v
     stream name: shotnum-channelid.bp          shotnum-channelid.s1.bp
```
This scenario is currently not implemented fully.

Example commands are as follows:
```
python generator.py --config config-jychoi.json
python receiver.py --config config-jychoi.json
```

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


# 2-node scenario
Data is streamed from a KSTAR DTN directly into a cori compute node.
```
KSTAR DTN                          cori compute

generator.py       =========>       processor_???.py
                       |
                       v
        ADIOS2 dataman
        stream name: KSTAR_shotnum
```
In this scenario, the processor reads a configuration file, performs the analysis routines
defined in that file, and stores the data for subsequent analysis.


As of now, there is no common format for configuration file. Each processor and receiver has its own
format. 


### Currently implemented processors

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







