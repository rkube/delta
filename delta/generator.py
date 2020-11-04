# -*- coding: UTF-8 -*-

from mpi4py import MPI
import sys

sys.path.append("/home/rkube/software/gcc/8.3/adios2/lib/python3.8/site-packages")
from os import path
import numpy as np

import json
import yaml
import argparse

import time

import logging, logging.config

from streaming.writers import writer_gen
from sources.dataloader import get_loader
from data_models.helpers import gen_channel_name, gen_var_name


"""
Author: R. Kube

Reads diagnostic data and stages it chunk-wise for transport.
"""


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Reads diagnostic data and stages it chunk-wise for transport.")
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

logger.info(f"Starting up...")

# Instantiate a dataloader
dataloader = get_loader(cfg)
logger.info(f"Creating writer_gen: engine={cfg['transport_nersc']['engine']}")

writer = writer_gen(cfg["transport_nersc"], gen_channel_name(cfg["diagnostic"]))
logger.info(f"Streaming channel name = {gen_channel_name(cfg['diagnostic'])}")
# Give the writer hints on what kind of data to transfer

writer.DefineVariable(gen_var_name(cfg)[rank],
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

writer.writer.Close()
logger.info(writer.transfer_stats())
logger.info("Finished")

# End of file generator.py
