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

For example within an interactive session, using a split terminal (see the screen utility)

# Workflow Scenario #1 (2-node scenario)
...

# Workflow Scenario #2 (3-node scenario)
It consists of three components:
```
  generator.py         =====>    receiver.py         =====>  analysis.py
(running on KSTAR DTN)   |     (running on NERSC DTN)  |      (running on NERSC compute nodes)
                         v                             v
     stream name: shotnum-channelid.bp          shotnum-channelid.s1.bp
```

Example commands are as follows:
```
python generator.py --config config-jychoi.json
python receiver.py --config config-jychoi.json
```

Parameters can be provided with a Jason file. Here is an example:
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
