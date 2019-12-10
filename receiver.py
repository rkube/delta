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

parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
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


## jyc: temporarily disabled. Will use later
#backend = mongo_backend(rank, my_channel_list)

#print("Starting main loop")

## jyc: 
## This is for testing to demontrate performing analysis as we receive data.
## We run multiple threads (or processes) to perform analysis for the data received from the generator (KSTAR).

## Testing between thread pool or process pool.
## Thread pool would be good for small number of workers and io-bound jobs.
## Processs pool would be good to utilize multiple cores.

#executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)

def perform_analysis(channel_data, step):
    """ 
    Perform analysis
    """ 
    print (">>> analysis ... %d"%step)
    t0 = time.time()
    if(my_analysis["name"] == "power_spectrum"):
        analysis_result = power_spectrum(channel_data, **my_analysis["config"])
    t1 = time.time()

    # Store result in database
    ##backend.store(my_analysis, analysis_result)
    #time.sleep(5)
    print (">>> analysis ... %d: done (%f secs)"%(step, t1-t0))

## Warming up for loading modules
print ("Warming up ... ")
for i in range(8):
    channel_data = np.zeros((num_channels, 100), dtype=np.float64)
    executor.submit(perform_analysis, channel_data, -1)
time.sleep(10)
print ("Warming up ... done")
    
## jyc:
## We run a worker thread to save the channel data received from the generator (KSTAR).
## Queue is used between the main process and this worker thread.
dq = queue.Queue()
def save_data():
    """
    Save channel data with Adios
    """
    fname = "{0:05d}_ch{1:06d}.s1.bp".format(shotnr, gen_id)
    with adios2.open(fname, "w", engine_type=cfg["analysis_engine"]) as fh:
        while True:
            channel_data, step = dq.get()
            if channel_data is None:
                break
            shape = channel_data.shape
            offset = [0,]*channel_data.ndim
            count = channel_data.shape
            fh.write("floats", channel_data, shape, offset, count, end_step=True)
            dq.task_done()
            print (">>> saving ... %d"%step)

worker = threading.Thread(target=save_data)
worker.start()

#reader = reader_dataman(shotnr, gen_id)
## general reader. engine type and params can be changed with the config file
reader = reader_gen(shotnr, gen_id, cfg["engine"], cfg["params"])
reader.Open()

## jyc:
## main loop: 
## Fetching data as soon as possible. Saving with Adios will be done by a thread.

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
        #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
    else:
        print("rank {0:d}: End of stream".format(rank))
        break

    # Recover channel data 
    channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    print ("Step begins ... %d"%step)
    ## jyc: this is just for testing. This is a place to run analysis if we want.
    executor.submit(perform_analysis, channel_data, step)

    ## Save data in a queue so that a workder thead will fetch and save concurrently.
    dq.put((channel_data, step))
    step += 1

#datamanReader.Close()
## jyc: this is just for testing. We need to close thread/process pool
executor.shutdown(wait=True)

## jyc:
## We are done. Wait the workder thread to finish.
dq.join()
dq.put((None, step))
worker.join()

print (">>> processing ... done.")

# End of file processor_adios2.
