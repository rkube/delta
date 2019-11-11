#-*- coding: UTF-8 -*-

"""
This processor implements the one-to-one model.

The processor runs as a single-task program and receives data from a single
Dataman stream. 
The configuration file defines a list of analysis routines together with a list of channel
data on which to apply a given routine. 
In the receive loop, this data is gathered and dispatched into a task queue


The task queue is implemented using Dask. Run this implementation within an interactive session.
Documentation on how to run with dask on nersc is here: https://docs.nersc.gov/programming/high-level-environments/python/dask/


On an interactice node:
1. Start the dask scheduler:
python -u $(which dask-scheduler) --scheduler-file $SCRATCH/scheduler.json

2. Start the worker tasks
srun -u -n 10 python -u $(which dask-worker) --scheduler-file $SCRATCH/scheduler.json --nthreads 1 &

3. Launch the dask program:
python -u processor_dask.py

"""

import numpy as np 
import adios2
import json
import argparse


from dask.distributed import Client
from processors.readers import reader_bpfile
from analysis.spectral import power_spectrum

from backends.mongodb import mongodb_backend
from analysis.spectral import power_spectrum


def analyze_and_store(channel_data, my_analysis, backend):
    """Analyze and store data

    channel_data:  ndarray, float: data to be analyzed
    method: string, name of the analysis method
    backend: obj, callable: storage backend

    """

    #print("Analyze and store")

    if my_analysis["name"] == "power_spectrum":
        result = power_spectrum(channel_data, **my_analysis["config"])

    backend.store(my_analysis, result)



dask_client = Client(scheduler_file="/global/cscratch1/sd/rkube/scheduler.json")

parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a dask queue")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# "Enforce" 1:1 mapping of reader processes on analysis tasks
assert(len(cfg["channel_lists"]) == size)
assert(len(cfg["analysis"]) == size)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
my_analysis = cfg["analysis"][rank]
my_channel_list = cfg["channel_lists"][rank]
gen_id = 100000 * rank + my_channel_list[0]
num_channels = len(my_channel_list)

reader = reader_bpfile(shotnr, gen_id)
reader.Open()


backend = mongodb_backend(rank, my_channel_list)

#print("Starting main loop")

while(True):
    stepStatus = reader.BeginStep()
    #print(stepStatus)
    if stepStatus == adios2.StepStatus.OK:
        #var = dataman_IO.InquireVariable("floats")
        #shape = var.Shape()
        #io_array = np.zeros(np.prod(shape), dtype=np.float)
        #reader.Get(var, io_array, adios2.Mode.Sync)
        channel_data = reader.get_data("floats")
        #currentStep = reader.CurrentStep()
        reader.EndStep()
        #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
    else:
        print("rank {0:d}: End of stream".format(rank))
        break

    # Recover channel data 
    channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    print(type(channel_data), channel_data.shape)

    dask_client.map(analyze_and_store, [channel_data, my_analysis, backend])
    #analyze_and_store(channel_data, my_analysis, backend)


    # Perform the analysis
    #if(my_analysis["name"] == "power_spectrum"):
    #    analysis_result = power_spectrum(channel_data, **my_analysis["config"])

    # Store result in database
    #backend.store(my_analysis, analysis_result)

#datamanReader.Close()


# End of file processor_adios2.