# -*- coding: UTF-8 -*-

from mpi4py import MPI
from os import path
import time
import numpy as np

import adios2
import json
import yaml
import argparse
import time

import logging, logging.config

from analysis.channels import channel_range
from streaming.writers import writer_gen
from streaming.adios_helpers import gen_channel_name_v2
from sources.loader_h5 import loader_h5

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

with open('configs/logger.yaml', 'r') as f:
    log_cfg = yaml.safe_load(f.read())
logging.config.dictConfig(log_cfg)

logger = logging.getLogger("simple")

datapath = cfg["transport"]["datapath"]
nstep = cfg["transport"]["nstep"]
shotnr = cfg["shotnr"]


# Enforce 1:1 mapping of channels and tasks
assert(len(cfg["transport"]["channel_range"]) == size)
# Channels this process is reading
ch_rg = channel_range.from_str(cfg["transport"]["channel_range"][rank])


# Generate a generator id from the MPI rank and the channel number of its first channel
#gen_id = 100_000 * rank + my_channel_range.ch_start.ch_num

# Use fluctana way to generate a channel name
#ch_name = gen_channel_name(shotnr, channel_id, rank)

# Use the channel_range.to_str() method to generate the channel name.
#ch_name = gen_channel_name_v2(shotnr, my_channel_range.to_str())
#logger.info(f"Rank: {rank:d}, channel_range: {my_channel_range}, ADIOS channel id = {ch_name}")

# Hard-code the total number of data points
data_pts = int(5e6)
# Hard-code number of data points per data packet
data_per_batch = int(1e1)
# Calculate the number of required data batches we send over the channel
num_batches = data_pts // data_per_batch

# Get a data_loader
dl = loader_h5(path.join(datapath, "ECEI.018431.LFS.h5"), ch_rg, cfg["transport"]["chunk_size"])

# Trying to load all h5 data into memory
logger.info("Loading h5 data into memory")
data_all = list()
for i in range(nstep):
    # dl.get return a list of an array of uint16 data
    # We convert as double
    data_arr = np.array(dl.get()).astype(np.float64)
    data_all.append(data_arr)

logger.info(f"Creating writer_gen: shotnr={shotnr}, engine={cfg['transport']['engine']}")

writer = writer_gen(cfg)
writer.DefineVariable(ch_rg.to_str(), data_arr)
writer.Open()

logger.info("Start sending:")
t0 = time.time()
for i in range(nstep):
    if(rank == 0):
        logger.info(f"Sending: {i:d} / {nstep:d}")
    writer.BeginStep()
    writer.put_data(data_all[i])
    writer.EndStep()
    #time.sleep(1.0)
t1 = time.time()
writer.writer.Close()

chunk_size = np.prod(data_arr.shape)*data_arr.itemsize/1024/1024
logger.info("")
logger.info("Summary:")
logger.info(f"    chunk shape: {data_arr.shape}")
logger.info(f"    chunk size (MB): {chunk_size:.03f}")
logger.info(f"    total nstep: {nstep:d}")
logger.info(f"    total data (MB): {(chunk_size*nstep):03f}")
logger.info(f"    time (sec): {(t1-t0):.03f}")
logger.info(f"    throughput (MB/sec): {(chunk_size*nstep)/(t1-t0):.03f}")

logger.info("Finished")

# End of file generator.py
