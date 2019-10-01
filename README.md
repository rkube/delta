DELTA-FUSION
aDaptive rEaL Time Analysis of big fusion data

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
srun -n 2 -c 2 --mem=1G --gres=craynetwork:0 python generator_adios2.py
srun -n 2 -c 2 --mem=1G --gres=craynetwork:0 python processor_adios2.py

For example within an interactive session, using a split terminal (see the screen utility)

