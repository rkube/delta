# -*- coding: UTF-8 -*-

from mpi4py import MPI
import numpy as np
import time
import adios2

from os import path

import json
import argparse

from streaming.writers import writer_gen
from fluctana import *

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


channels = expand_clist(cfg["channel_range"])
# #DON'T USE MPI to split channels now, use for timechunks instead
# n = len(channels) // size
# channel_split = [channels[i:i+n] for i in range(0,len(channels),n)]
# # Enforce 1:1 mapping of channels and tasks
# assert(len(channel_split) == size)
# # Channels this process is reading
# my_channel_range = channel_split[rank]
# gen_id = 100000 * rank# + my_channel_range[0] #TODO: WIll this matter witout my_channel_range?

# print("Rank: {0:d}".format(rank), ", channel_range: ", my_channel_range, ", id = ", gen_id)


# Get a data_loader
dobj = KstarEcei(shot=shotnr,data_path=datapath,clist=channels,verbose=False)
cfg.update({'TriggerTime':dobj.tt.tolist(),'SampleRate':[dobj.fs/1e3],
            'TFcurrent':dobj.itf/1e3,'Mode':dobj.mode, 
            'LoFreq':dobj.lo,'LensFocus':dobj.sf,'LensZoom':dobj.sz})



# Hard-code the total number of data points
data_pts = int(5e6)
# Read number of data points per data packet from cfg
batch_size = cfg['batch_size'] #int(1e4)
# Calculate the number of required data batches we send over the channel. Make sure to round up (ceil).
num_batches = np.ceil(data_pts / batch_size).astype(int)

batch_idxs = np.array_split(np.arange(num_batches),size)
num_batches_rank = batch_idxs[rank].size


# Trying to load all h5 data into memory
print("Loading h5 data into memory")
timebase = dobj.time_base_full()
tstarts = timebase[::batch_size]
tstops = timebase[batch_size-1::batch_size]
tstartminrank = tstarts[batch_idxs[rank][0]]
tstopmaxrank = tstops[batch_idxs[rank][-1]]
_,data = dobj.get_data(trange=[tstartminrank,tstopmaxrank],norm=1,verbose=0)
data_all = np.array_split(data,num_batches_rank,axis=-1)

######From this point, this is the same as generator_brute.py. The data_all
######is a list of the data chunks to send over, and each rank has a 
######subset of the total number of chunks
######To be more like a true streaming from diagnostic scenario, we could
######switch to interleaving, having each rank read round-robin read the
######the data chunks, but with the added overhead of disk touches, this
######would slow us down, more than the true streaming scenario

#writer = writer_dataman(shotnr, gen_id)
writer = writer_gen(shotnr, gen_id, cfg["engine"], cfg["params"])

data_arr = data_all[0]
varData = writer.DefineVariable("floats",data_arr)
varTime = writer.DefineVariable("trange",np.array([0.0,0.0]))
writer.DefineAttributes("cfg",cfg)
writer.Open()

print("Start sending:")
t0 = time.time()
for i in range(nstep):
    if(rank == 0):
        print("Sending: {0:d} / {1:d}".format(i, nstep))
    writer.BeginStep()
    writer.put_data(varTime,np.array([tstarts[i],tstops[i]]))
    writer.put_data(varData,data_all[i])
    writer.EndStep()
t1 = time.time()
writer.writer.Close()

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

