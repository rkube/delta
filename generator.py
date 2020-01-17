# -*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np
import time
import adios2

from os import path

import json
import argparse

from generators.writers import writer_dataman, writer_bpfile, writer_sst, writer_gen
from generators.data_loader import data_loader

"""
Generates batches of ECEI data.
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='config.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
nstep = cfg["nstep"]

# Enforce 1:1 mapping of channels and tasks
assert(len(cfg["channel_range"]) == size)
# Channels this process is reading
my_channel_range = cfg["channel_range"][rank]
gen_id = 100000 * rank + my_channel_range[0]

print("Rank: {0:d}".format(rank), ", channel_range: ", my_channel_range, ", id = ", gen_id)

# Hard-code the total number of data points
data_pts = int(5e6)
# Hard-code number of data points per data packet
data_per_batch = int(1e1)
# Calculate the number of required data batches we send over the channel
num_batches = data_pts // data_per_batch

# Get a data_loader
dl = data_loader(path.join(datapath, "ECEI.018431.LFS.h5"),
                 channel_range=my_channel_range,
                 batch_size=cfg["batch_size"])

# Trying to load all h5 data into memory
print("Loading h5 data into memory")
data_all = list()
for i in range(nstep):
    # dl.get return a list of an array of uint16 data
    # We convert as double
    data_arr = np.array(dl.get()).astype(np.float64)
    data_all.append(data_arr)

#writer = writer_dataman(shotnr, gen_id)
writer = writer_gen(shotnr, gen_id, cfg["engine"], cfg["params"])

writer.DefineVariable(data_arr)
writer.Open()

print("Start sending:")
t0 = time.time()
for i in range(nstep):
    if(rank == 0):
        print("Sending: {0:d} / {1:d}".format(i, nstep))
    writer.put_data(data_all[i])
t1 = time.time()

chunk_size = np.prod(data_arr.shape)*data_arr.itemsize/1024/1024
print("")
print("Summary:")
print("    chunk shape:", data_arr.shape)
print("    chunk size (MB): {0:.03f}".format(chunk_size))
print("    total nstep:", nstep)
print("    total data (MB): {0:.03f}".format(chunk_size*nstep))
print("    time (sec): {0:.03f}".format(t1-t0))
print("    throughput (MB/sec): {0:.03f}".format((chunk_size*nstep)/(t1-t0)))

print("Finished")

