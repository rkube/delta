# -*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np
import time
import adios2

from os import path

import json
import argparse

from streaming.writers import writer_dataman, writer_bpfile
from streaming.data_loader import data_loader

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

# Enforce 1:1 mapping of channels and tasks
assert(len(cfg["channel_ranges"]) == size)
# Channels this process is reading
my_channel_range = cfg["channel_ranges"][rank]
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
                 batch_size=1000)

# data_arr is a list
_data_arr = dl.get()
data_arr = np.array(_data_arr)
data_arr = data_arr.astype(np.float64)
_data_arr = 0.0

writer = writer_bpfile(shotnr, gen_id)
writer.DefineVariable(data_arr)
writer.Open()

for i in range(10):
    if(rank == 0):
        print("Sending: {0:d} / {1:d}".format(i, 10))
    writer.put_data(data_arr)
    dl.get()
    time.sleep(0.1)

#print("Finished")

