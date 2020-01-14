#-*- coding: UTF-8 -*-
# Example command: mpirun -n 8 python -u -m mpi4py.futures receiver_mpi.py --config config.json
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

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import sys

parser = argparse.ArgumentParser(description="Receive KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
parser.add_argument('--nompi', help='Use with nompi', action='store_true')
## A trick to handle: python -u -m mpi4py.futures ...
idx = len(sys.argv) - sys.argv[::-1].index(__file__)
args = parser.parse_args(sys.argv[idx:])

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

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
my_analysis = cfg["analysis"][0]
my_channel_list = cfg["channel_range"][0]
gen_id = 100000 * 0 + my_channel_list[0]
num_channels = len(my_channel_list)

# Function for workers to perform, which is an analysis.
# Workers (non-master MPI workers) will process each chunk of data.
# mpi4py will be responsible for data distribution.
def perform_analysis(channel_data, step):
    """ 
    Perform analysis
    """ 
    print(">>>         ({0:d}) Worker: do analysis step={1:d}".format(rank, step))
    t0 = time.time()
    if(my_analysis["name"] == "power_spectrum"):
        analysis_result = power_spectrum(channel_data, **my_analysis["config"])
    t1 = time.time()

    # Store result in database
    # backend.store(my_analysis, analysis_result)
    time.sleep(10)
    print(">>>         ({0:d}) Worker: done with analysis step={1:d} ({2:f} secs)".format(rank, step, t1-t0))

# Function for a helper thead (dispatcher).
# The dispatcher will dispatch data in the queue (dq) and 
# distribute to other workers (non-master MPI workers) with mpi4py's MPICommExecutor.
def dispatch():
    while True:
        channel_data, step = dq.get()
        print(">>>     ({0:d}) Dispatcher: read data step={1:d}".format(rank, step))
        if channel_data is None:
            break
        shape = channel_data.shape
        offset = [0,]*channel_data.ndim
        count = channel_data.shape
        future = executor.submit(perform_analysis, channel_data, step)
        dq.task_done()

# Main
if __name__ == "__main__":
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            # Only master will execute the following block
            # Use of "__main__" is critical

            # The master thread will keep reading data, while 
            # a helper thread (dispatcher) will dispatch jobs in the queue (dq) asynchronously 
            # and distribute jobs to other workers.
            # The main idea is not to slow down the master.
            dq = queue.Queue()
            dispatcher = threading.Thread(target=dispatch)
            dispatcher.start()

            # Only the master thread will open a data stream.
            # General reader: engine type and params can be changed with the config file
            reader = reader_gen(shotnr, gen_id, cfg["engine"], cfg["params"])
            reader.Open()

            # Main loop is here
            # Reading data (from KSTAR) and save in the queue (dq) as soon as possible.
            # Dispatcher (a helper thread) will asynchronously fetch data in the queue and distribute to other workers.
            while(True):
                stepStatus = reader.BeginStep()
                if stepStatus == adios2.StepStatus.OK:
                    channel_data = reader.get_data("floats")
                    currentStep = reader.CurrentStep()
                    reader.EndStep()
                    #print("rank {0:d}: Step".format(rank), reader.CurrentStep(), ", io_array = ", io_array)
                else:
                    print(">>> ({0:d}) Receiver: end of stream".format(rank))
                    break

                # Recover channel data 
                channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))
                print(">>> ({0:d}) Receiver: received data step={1:d}".format(rank, currentStep))

                # Save data in a queue then go back to work
                # Dispatcher (a helper thread) will fetch asynchronously.
                dq.put((channel_data, currentStep))
                time.sleep(1)

            ## Clean up
            dq.join()
            dq.put((None, -1))
            dispatcher.join()
            print(">>> ({0:d}) Receiver: done.".format(rank))

# End of file processor_adios2.
