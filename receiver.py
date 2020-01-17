#-*- coding: UTF-8 -*-

import numpy as np 
import adios2
import json
import argparse

from analysis.spectral import power_spectrum

import concurrent.futures
import time
import os
import queue
import threading

parser = argparse.ArgumentParser(description="Receive KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
parser.add_argument('--nompi', help='Use with nompi', action='store_true')
args = parser.parse_args()

if not args.nompi:
    from processors.readers import reader_dataman, reader_bpfile, reader_sst, reader_gen
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    from processors.readers_nompi import reader_dataman, reader_bpfile, reader_sst, reader_gen
    comm = None
    rank = 0
    size = 1

with open(args.config, "r") as df:
    cfg = json.load(df)
    df.close()

# "Enforce" 1:1 mapping of reader processes on analysis tasks
assert(len(cfg["channel_range"]) == size)
assert(len(cfg["analysis"]) == size)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
my_analysis = cfg["analysis"][rank]
my_channel_list = cfg["channel_range"][rank]
gen_id = 100000 * rank + my_channel_list[0]
num_channels = len(my_channel_list)
    
## jyc:
## We create N queues and N analysis workers. Each worker will be assigned to one queue.
## Each worker will extract data from a queue and write as Adios object.

num_analysis = int(cfg["num_analysis"])
print (">>> num_analysis: %d"%num_analysis)
def save_data(worker_id):
    """
    Save channel data with Adios
    """
    print (">>> worker %d: hello"%(worker_id))
    fname = "{0:05d}_ch{1:06d}.s{2:02d}.bp".format(shotnr, gen_id, worker_id)
    with adios2.open(fname, "w", engine_type=cfg["analysis_engine"]) as fh:
        while True:
            channel_data, step = queue_list[worker_id].get()
            if channel_data is None:
                break
            shape = channel_data.shape
            offset = [0,]*channel_data.ndim
            count = channel_data.shape
            print (">>> worker %d: saving ... %d"%(worker_id, step))
            fh.write("floats", channel_data, shape, offset, count, end_step=True)
            queue_list[worker_id].task_done()

queue_list = list()
worker_list = list()

for n in range(num_analysis):
    dq = queue.Queue()
    queue_list.append(dq)
    worker = threading.Thread(target=save_data, args=(n,))
    worker.start()
    worker_list.append(worker)

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
        print(">>> receiver {0:d}: End of stream".format(rank))
        break

    # Recover channel data 
    channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    print (">>> Step begins ... %d"%step)
    ## jyc: this is just for testing. This is a place to run analysis if we want.
    ##executor.submit(perform_analysis, channel_data, step)

    ## Save data in a queue so that a workder thead will fetch and save concurrently.
    queue_list[step%num_analysis].put((channel_data, step))
    step += 1

## jyc:
## We are done. Wait workder threads to finish.
for n in range(num_analysis):
    dq = queue_list[n]
    worker = worker_list[n]
    dq.join()
    dq.put((None, step))
    worker.join()

print (">>> receiver ... done.")

# End of file processor_adios2.
