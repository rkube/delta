# -*- coding: UTF-8 -*-

#import h5py
from mpi4py import MPI
import numpy as np
#import pickle
import time
import adios2

import json
import argparse

from writers import writer_dataman, writer_bpfile

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
channel_list = cfg["channels"]
assert(len(channel_list) == size)

# Later: Load K-Star data from HDF5
#df_fname = "{0:s}/{1:06d}/ECEI.{1:06d}.HFS.h5".format(datapath, shotnr, shotnr)

# Hard-code the total number of data points
data_pts = int(5e6)
# Hard-code number of data points per data packet
data_per_batch = int(1e1)
# Calculate the number of required data batches we send over the channel
num_batches = data_pts // data_per_batch


# # Data to be transported:
data_arr = np.zeros(data_per_batch, dtype=np.float)

#print("Initializing...")
writer = writer_bpfile(shotnr, channel_list[rank])
writer.DefineVariable(data_arr)
writer.Open()


#print("...done. Ready to stream")

for i in range(10):
    if(rank == 0):
        print("Sending: {0:d} / {1:d}".format(i, 10))
    data_arr = data_arr + 1.0
    writer.put_data(data_arr)
    time.sleep(0.1)

#print("Finished")

