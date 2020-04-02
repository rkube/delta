# -*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np
import time
import adios2

from os import path

import json
import argparse

from analysis.channels import channel_range
from generators.writers import writer_dataman, writer_bpfile, writer_sst, writer_gen
from generators.data_loader import data_loader

"""
Distributes time-chunked ECEI data via ADIOS2.
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/test_generator.json')
args = parser.parse_args()

with open(args.config, "r") as df:
    cfg = json.load(df)

datapath = cfg["datapath"]
shotnr = cfg["shotnr"]
nstep = cfg["nstep"]

# Enforce 1:1 mapping of channels and tasks
assert(len(cfg["channel_range"]) == size)
# Channels this process is reading
my_channel_range = channel_range.from_str(cfg["channel_range"][rank])
gen_id = f"{100000 * rank}{my_channel_range.dev}"

print(f"Rank: {rank:d}, channel_range: {my_channel_range}, ADIOS channel id = {gen_id}")

# Hard-code the total number of data points
data_pts = int(5e6)
# Hard-code number of data points per data packet
data_per_batch = int(1e1)
# Calculate the number of required data batches we send over the channel
num_batches = data_pts // data_per_batch

# Get a data_loader
dl = data_loader(path.join(datapath, "ECEI.018431.LFS.h5"),
                 my_channel_range, cfg["chunk_size"])

# Trying to load all h5 data into memory
print("Loading h5 data into memory")
data_all = list()
for i in range(nstep):
    # dl.get return a list of an array of uint16 data
    # We convert as double
    data_arr = np.array(dl.get()).astype(np.float64)
    data_all.append(data_arr)

writer = writer_gen(shotnr, gen_id, cfg["engine"], cfg["params"])
writer.DefineVariable(my_channel_range.to_str(), data_arr)
writer.Open()

print("Start sending:")
t0 = time.time()
for i in range(nstep):
    if(rank == 0):
        print(f"Sending: {i:d} / {nstep:d}")
    writer.BeginStep()
    writer.put_data(data_all[i])
    writer.EndStep()
t1 = time.time()

chunk_size = np.prod(data_arr.shape)*data_arr.itemsize/1024/1024
print("")
print("Summary:")
print(f"    chunk shape:", data_arr.shape)
print(f"    chunk size (MB): {chunk_size:.03f}")
print(f"    total nstep: {nstep:d}")
print(f"    total data (MB): {(chunk_size*nstep):03f}")
print(f"    time (sec): {(t1-t0):.03f}")
print(f"    throughput (MB/sec): {(chunk_size*nstep)/(t1-t0):.03f}")

print("Finished")

# End of file generator.py