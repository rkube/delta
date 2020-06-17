# -*- coding: UTF-8 -*-

from mpi4py import MPI
from os import path
import numpy as np

import adios2
import json
import yaml
import argparse

import timeit
import time

import logging, logging.config

from analysis.channels import channel_range
from streaming.writers import writer_gen
from sources.loader_ecei_cached import loader_ecei

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

logger = logging.getLogger("generator")

datapath = cfg["transport_nersc"]["datapath"]
nstep = cfg["transport_nersc"]["nstep"]
shotnr = cfg["shotnr"]

# Enforce 1:1 mapping of channels and tasks
assert(len(cfg["transport_nersc"]["channel_range"]) == size)
# Channels this process is reading
ch_rg = channel_range.from_str(cfg["transport_nersc"]["channel_range"][rank])

# Hard-code the total number of data points
#data_pts = int(5e6)
# Hard-code number of data points per data packet
#data_per_batch = int(1e1)
# Calculate the number of required data batches we send over the channel
#num_batches = data_pts // data_per_batch

# Get a data_loader
logger.info("Loading h5 data into memory")
dl = loader_ecei(cfg)
dl.cache()
batch_gen = dl.batch_generator()

logger.info(f"Creating writer_gen: shotnr={shotnr}, engine={cfg['transport_nersc']['engine']}")

writer = writer_gen(cfg["transport_nersc"])

# Pass data layout to writer and reset generator
for data in batch_gen:
    break
writer.DefineVariable(ch_rg.to_str(), data)
batch_gen = dl.batch_generator()

writer.Open()

logger.info("Start sending on channel:")
tic = timeit.default_timer()
nstep = 0
for data in batch_gen:
    if(rank == 0):
        logger.info(f"Sending time_chunk {nstep} / {dl.num_chunks}")
    writer.BeginStep()
    writer.put_data(data)
    writer.EndStep()
    nstep += 1
    time.sleep(0.1)

toc = timeit.default_timer()
writer.writer.Close()

chunk_size = np.prod(data.shape) * data.itemsize / 1024 / 1024
logger.info("")
logger.info("Summary:")
logger.info(f"    chunk shape: {data.shape}")
logger.info(f"    chunk size (MB): {chunk_size:.03f}")
logger.info(f"    total nstep: {nstep:d}")
logger.info(f"    total data (MB): {(chunk_size * nstep):03f}")
logger.info(f"    time (sec): {(toc - tic):.03f}")
logger.info(f"    throughput (MB/sec): {(chunk_size * nstep)/(toc - tic):.03f}")

logger.info("Finished")

# End of file generator.py
