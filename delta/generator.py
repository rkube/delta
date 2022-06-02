# -*- coding: UTF-8 -*-

"""Stages diagnostic data for streaming.

Reads diagnostic data and stages it chunk-wise for transport.
Data stream can be received with middleman or processor.
"""

from mpi4py import MPI

import json
import yaml
import argparse

import time

import logging
import logging.config

from streaming.writers import writer_gen
from sources.helpers import get_loader
from data_models.helpers import gen_channel_name, gen_var_name

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Reads diagnostic data and stages" +
                                 " it chunk-wise for transport.")
parser.add_argument('--config', type=str,
                    help='Lists the configuration file',
                    default='configs/test_generator.json')
parser.add_argument('--kstar', help='kstar', action='store_true')
parser.add_argument('--slow', type=float,
                    help="Adds a small break in-between sending packages.",
                    default=0.1)
args = parser.parse_args()

# set up the configuration
with open(args.config, "r") as df:
    cfg = json.load(df)

# Set up the logger
with open('configs/logger.yaml', 'r') as f:
    log_cfg = yaml.safe_load(f.read())
logging.config.dictConfig(log_cfg)
logger = logging.getLogger("generator")
logger.info("Starting up...")

# Instantiate a dataloader
dataloader = get_loader(cfg)
sectionname = "transport_tx" if not args.kstar else "transport_rx"
logger.info(f"Creating writer_gen: engine={cfg[sectionname]['engine']}")

writer = writer_gen(cfg[sectionname], gen_channel_name(cfg["diagnostic"]))
logger.info(f"Streaming channel name = {gen_channel_name(cfg['diagnostic'])}")
# Give the writer hints on what kind of data to transfer

writer.DefineVariable(gen_var_name(cfg)[rank],
                      dataloader.get_chunk_shape(),
                      dataloader.dtype)
# TODO: Clean up naming conventions for stream attributes
logger.info(f"Writing attributes: {dataloader.attrs}")

writer.Open()
writer.DefineAttributes("stream_attrs", dataloader.attrs)

logger.info("Start sending on channel:")
batch_gen = dataloader.batch_generator()
for nstep, chunk in enumerate(batch_gen):
    # TODO: Do we want to place filtering in the generator? This would allow us to
    # send fewer data to the processor.
    # if rank == 0:
    #     logger.info(f"Filtering time_chunk {nstep} / {dataloader.num_chunks}")

    if rank == 0:
        logger.info(f"Sending time_chunk {nstep} / {dataloader.num_chunks}")
    writer.BeginStep()
    writer.put_data(gen_var_name(cfg)[rank],chunk)
    writer.EndStep()
    time.sleep(args.slow)


writer.writer.Close()
logger.info(writer.transfer_stats())
logger.info("Finished")

# End of file generator.py
