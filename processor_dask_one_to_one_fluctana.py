#-*- coding: UTF-8 -*-

"""
This processor implements the one-to-one model. Code is adapted from processor_dask_one_to_one.py
to use the kstar fluctana object class.

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
import dask.array as da

import json
import argparse
from distributed import Client, progress

from backends.mongodb import mongodb_backend
from readers.reader_one_to_one import reader_bpfile
from analysis.task_fft import task_fft_scipy

from analysis.task_spectral import task_spectral, task_cross_phase


# task_object_dict maps the string-value of the analysis field in the json file
# to an object that defines an appropriate analysis function.
task_object_dict = {"cross_phase": task_cross_phase}



# This object manages storage to a backend.
mongo_client = mongodb_backend()
# Interface to worker nodes
dask_client = Client(scheduler_file="/global/cscratch1/sd/rkube/scheduler.json")


# Add the source path to all workers so that the imports are working :)
def add_path():
    import sys
    sys.path.append("/global/homes/r/rkube/repos/delta")

dask_client.run(add_path)


# Parse command line arguments and read configuration file
parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a dask queue")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config_one_to_one_fluctana.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# Sample rate in Hz
cfg["fft_params"]["fsample"] = cfg["ECEI_cfg"]["SampleRate"] * 1e3


# Build list of analysis tasks that are performed at any given time step
# Here we iterate over the task list defined in the json file
# Each task needs to define the field 'analysis' that describes an analysis to be performed
# The dictionary task_object_dict maps the string value of this field to an object that is later
# called to perform the analysis.
task_list = []
for task_config in cfg["task_list"]:
    task_list.append(task_object_dict[task_config["analysis"]](task_config))

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
reader = reader_bpfile(shotnr)
reader.Open(cfg["datapath"])

# Create the FFT task
my_fft = task_fft_scipy(10_000, cfg["fft_params"], normalize=True, detrend=True)


print("Starting main loop")
s = 0

while(True):
    stepStatus = reader.BeginStep()

    # Iterate over the task list and update the required data at the current time step
    if stepStatus:
        print("ok")
        task_futures = []

        # generate a dummy time-base for the data of the current chunk
        dummy_tb = np.arange(0.0, 2e-2, 2e-6) * float(s+1)
  
        # Get the raw data from the stream
        stream_data = reader.Get()
        # Option 1) Create a dask array
        # See http://matthewrocklin.com/blog/work/2017/01/17/dask-images
        #raw_data = da.from_array(raw_data, chunks=(1, 10_000))
        #dask_client.persist(raw_data)

        # Option 2)
        # Scatter the raw data to the workers. To reference the t
        stream_data_future = dask_client.scatter(stream_data, broadcast=True)
        print("*** types(stream_data_future), ", type(stream_data_future), stream_data_future)

        # Perform a FFT on the raw data
        fft_future = my_fft.do_fft(dask_client, stream_data_future)
        # gather pulls the result of the operation: 
        # https://docs.dask.org/en/latest/futures.html#distributed.Client.gather
        results = dask_client.gather(fft_future)

        #print(type(results))
        #for r in results:
        #    print("***main: type(r): ", type(r))
        fft_data = da.from_array(results)
        #dask_client.persist(fft_data)
        fft_future = dask_client.scatter(fft_data, broadcast=True)
        
        #if (s == 0):
        #    print("***storing fft_data: ", type(fft_data.compute()))
        #    np.savez("dask_fft_data_s{0:04d}.npz".format(s), fft_data=fft_data.compute())


        print("*** type(fft_future) = ", type(fft_future))

        for task in task_list:
            task.calculate(dask_client, fft_future)
            #task.update_data(data_ft, dummy_tb)

            # Method 1: Pass dask_client to object
            #task_futures.append(task.method(dask_client))

            # Method 2: Get method and data from object
            #task_futures.append(dask_client.submit(task.get_method(), task.get_data()))

    else:
        print("End of stream")
        break

    reader.EndStep()

    # Pass the task object to our backend for storage
    for task in task_list:
        #print(task.futures_list)
         mongo_client.store(task, dummy=True)

    # Do only 10 time steps for now
    s -= -1
    if s >= 2:
        break


# End of file processor_dask_one_to_one.py.