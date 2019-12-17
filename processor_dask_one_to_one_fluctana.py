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
#import dask.array as da

import json
import argparse
from distributed import Client, progress

#from backends.mongodb import mongodb_backend
from backends.backend_numpy import backend_numpy
from readers.reader_one_to_one import reader_bpfile
from analysis.task_fft import task_fft_scipy

from analysis.task_spectral import task_spectral, task_cross_phase, task_cross_power, task_coherence, task_bicoherence, task_xspec, task_cross_correlation, task_skw

import timeit


# task_object_dict maps the string-value of the analysis field in the json file
# to an object that defines an appropriate analysis function.
task_object_dict = {"cross_phase": task_cross_phase,
                    "cross_power": task_cross_power,
                    "coherence": task_coherence,
                    "bicoherence": task_bicoherence,
                    "xspec": task_xspec,
                    "cross_correlation": task_cross_correlation,
                    "skw": task_skw}


# This object manages storage to a backend.
#mongo_client = mongodb_backend()
# Interface to worker nodes
dask_client = Client(scheduler_file="/global/cscratch1/sd/rkube/scheduler.json")


# TODO: Add this to a pre-loeading
# Add the source path to all workers so that the imports are working :)
def add_path():
    import sys
    import numpy as np
    sys.path.append("/global/homes/r/rkube/repos/delta")
#dask_client.run(add_path)

store_backend = backend_numpy("/global/homes/r/rkube/repos/delta/test_data")

# Parse command line arguments and read configuration file
parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a dask queue")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config_one_to_one_fluctana.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# Sample rate in Hz
cfg["fft_params"]["fsample"] = cfg["ECEI_cfg"]["SampleRate"] * 1e3

# Create the FFT task
my_fft = task_fft_scipy(10_000, cfg["fft_params"], normalize=True, detrend=True)
fft_params = my_fft.get_fft_params()

# Build list of analysis tasks that are performed at any given time step
# Here we iterate over the task list defined in the json file
# Each task needs to define the field 'analysis' that describes an analysis to be performed
# The dictionary task_object_dict maps the string value of this field to an object that is later
# called to perform the analysis.
task_list = []
for task_config in cfg["task_list"]:
    task_list.append(task_object_dict[task_config["analysis"]](task_config, fft_params, cfg["ECEI_cfg"]))
    task_list[-1].store_metadata(store_backend)

reader = reader_bpfile(cfg["shotnr"], cfg["ECEI_cfg"])
reader.Open(cfg["datapath"])


do_local_fft = True
print("Starting main loop")
s = 0

while(True):
    stepStatus = reader.BeginStep()
    tic_tstep = timeit.default_timer()

    # Iterate over the task list and update the required data at the current time step
    if stepStatus:
        print("ok")
        task_futures = []

        # generate a dummy time-base for the data of the current chunk
        dummy_tb = np.arange(0.0, 2e-2, 2e-6) * float(s+1)
  
        # Get the raw data from the stream
        stream_data = reader.Get()
        tb = reader.gen_timebase()
        
        if do_local_fft:
            tic_fft = timeit.default_timer()
            fft_data = my_fft.do_fft_local(stream_data)
            toc_fft = timeit.default_timer()
        else:
            # Perform a FFT on the raw data
            tic_fft = timeit.default_timer()
            stream_data_future = dask_client.scatter(stream_data, broadcast=True)
            fft_future = my_fft.do_fft(dask_client, stream_data_future)
            # gather pulls the result of the operation: 
            # https://docs.dask.org/en/latest/futures.html#distributed.Client.gather
            results = dask_client.gather(fft_future)
            toc_fft = timeit.default_timer()
            fft_data = np.array(results)
    
        print("*** main_loop: FFT took {0:f}s".format(toc_fft - tic_fft))
        
        np.savez("test_data/fft_data_s{0:04d}.npz".format(s), fft_data=fft_data)

        # Broadcast the fourier-transformed data to all workers
        # Pas this future to worker tasks as a reference to fft_data
        tic_sc = timeit.default_timer()
        fft_future = dask_client.scatter(fft_data, broadcast=True)
        toc_sc = timeit.default_timer()
        print("*** main_loop scatter(fft_data) took {0:f}s.".format(toc_sc - tic_sc))


        tic_task = timeit.default_timer()
        for task in task_list:
            print("*** main_loop: ", task.description)
            task.calculate(dask_client, fft_future)
            #task.update_data(data_ft, dummy_tb)

            # Method 1: Pass dask_client to object
            #task_futures.append(task.method(dask_client))

            # Method 2: Get method and data from object
            #task_futures.append(dask_client.submit(task.get_method(), task.get_data()))

        toc_task = timeit.default_timer()
        print("*** main_loop: dispatching tasks took {0:6.4f}s".format(toc_task - tic_task))

    else:
        print("End of stream")
        break

    reader.EndStep()

    # Store result in npz file for quick access
    store_backend.ctr = s
    tic_store = timeit.default_timer()
    for task in task_list:
        task.store_data(store_backend, {"tstep": s})
    toc_store = timeit.default_timer()

    print("*** main_loop store_data took {0:6.4f}s.".format(toc_store - tic_store))

    #tic_mongo = timeit.default_timer()
    # Pass the task object to our backend for storage
    #for task in task_list:
    #     mongo_client.store(task, dummy=True)
    #toc_mongo = timeit.default_timer()
    #print("Storing data: Elapsed time: ", toc_mongo - tic_mongo)

    toc_tstep = timeit.default_timer()
    print("Processed timestep {0:d}: Elapsed time: {1:6.3f}s".format(s, toc_tstep - tic_tstep))

    # Do only 10 time steps for now
    s -= -1
    if s > 2:
        break


# End of file processor_dask_one_to_one.py.