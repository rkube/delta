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

#from analysis.channels import channel_range
from streaming.writers import writer_gen
from sources.dataloader import get_loader

"""
Distributes time-chunked ECEI data via ADIOS2.
"""


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/test_generator.json')
args = parser.parse_args()

# set up the configuration
with open(args.config, "r") as df:
    cfg = json.load(df)

# Set up the logger
with open('configs/logger.yaml', 'r') as f:
    log_cfg = yaml.safe_load(f.read())
logging.config.dictConfig(log_cfg)
logger = logging.getLogger("generator")

# Instantiate a dataloader
dataloader  = get_loader(cfg)

logger.info(f"Creating writer_gen: engine={cfg['transport_nersc']['engine']}")

writer = writer_gen(cfg["transport_nersc"], dataloader.stream_name)

# Give the writer hints on what kind of data to transfer
writer.DefineVariable(dataloader.get_channel_name(), 
                      dataloader.get_chunk_shape(),
                      dataloader.dtype)
writer.Open()

logger.info("Start sending on channel:")

batch_gen = dataloader.batch_generator()
for nstep, chunk in enumerate(batch_gen):
    if rank == 0:
        logger.info(f"Filtering time_chunk {nstep} / {dataloader.num_chunks}")

    if rank == 0:
        logger.info(f"Sending time_chunk {nstep} / {dataloader.num_chunks}")
    writer.BeginStep()
    writer.put_data(chunk, {"tidx": nstep})
    writer.EndStep()
    time.sleep(0.1)

    if nstep > 10:
        break

writer.writer.Close()
logger.info(writer.transfer_stats())
logger.info("Finished")

# End of file generator.py