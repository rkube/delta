#-*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np 
import adios2
import json
import argparse

from processors.readers import reader_dataman, reader_bpfile, reader_sst, reader_gen
from analysis.spectral import power_spectrum

## jyc: temporarily disabled. Will use later
#from backends.mongodb import mongo_backend

import concurrent.futures
import time
import os
import queue
import threading

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="Perform analysis")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# "Enforce" 1:1 mapping of reader processes on analysis tasks
#assert(len(cfg["channel_range"]) == size)
#assert(len(cfg["analysis"]) == size)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
my_analysis = cfg["analysis"][0]
my_channel_list = cfg["channel_range"][0]
gen_id = 100000 * 0 + my_channel_list[0]
num_channels = len(my_channel_list)


## jyc: temporarily disabled. Will use later
#backend = mongo_backend(rank, my_channel_list)

## jyc: 
## We run multiple threads (or processes) to perform analysis for the data received from the receiver (NERSC).

## Testing between thread pool or process pool.
## Thread pool would be good for small number of workers and io-bound jobs.
## Processs pool would be good to utilize multiple cores.

#executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)

def perform_analysis(channel_data, step):
    """ 
    Perform analysis
    """ 
    print (">>> analysis rank %d: analysis ... %d"%(rank, step))
    t0 = time.time()
    if(my_analysis["name"] == "power_spectrum"):
        analysis_result = power_spectrum(channel_data, **my_analysis["config"])
    t1 = time.time()

    # Store result in database
    ##backend.store(my_analysis, analysis_result)
    #time.sleep(5)
    print (">>> analysis rank %d: analysis ... %d: done (%f secs)"%(rank, step, t1-t0))

## Warming up for loading modules
print (">>> analysis rank %d: Warming up ... "%rank)
for i in range(8):
    channel_data = np.zeros((num_channels, 100), dtype=np.float64)
    executor.submit(perform_analysis, channel_data, -1)
time.sleep(10)
print (">>> analysis rank %d: Warming up ... done"%rank)
    
#reader = reader_dataman(shotnr, gen_id)
## general reader. engine type and params can be changed with the config file
reader = reader_gen(shotnr, gen_id, cfg["analysis_engine"], cfg["analysis_engine_params"])
reader.Open(worker_id=rank)

## jyc:
## main loop: 
## Fetching data as soon as possible and call workers in the thread pool (or process pool)

step = 0
while(True):
#for i in range(10):
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
    else:
        print(">>> analysis rank %d: End of stream"%(rank))
        break

    # Recover channel data 
    channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    print (">>> analysis rank %d: Step begins ... %d"%(rank, step))
    ## jyc: this is just for testing. This is a place to run analysis if we want.
    executor.submit(perform_analysis, channel_data, step)

    step += 1

#datamanReader.Close()
## jyc: this is just for testing. We need to close thread/process pool
executor.shutdown(wait=True)

print (">>> analysis rank %d: processing ... done."%rank)

# End of file processor_adios2.
