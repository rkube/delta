#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""Uses readers in multi-channel setting

Command

```
$ srun -n NN python test_reader_multichannel.py
"""

from mpi4py import MPI
import adios2
import numpy as np

import logging
import time


import sys
sys.path.append("/global/homes/r/rkube/repos/delta/delta")
from streaming.reader_mpi import reader_gen
from streaming.adios_helpers import gen_channel_name
from streaming.stream_stats import stream_stats
from data_models.base_models import twod_chunk


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logging.basicConfig(level=20,
                    format=f"rank {rank:1d} - " + " %(asctime)15s - %(levelname)s - %(message)s")

# Define transport section for transport
cfg_transport = {
    "datapath": "tmp_nersc",
    "engine": "dataman",
    "params":
    {
      "IPAddress": "203.230.120.125",
      "Timeout": "120",
      "Port": "50001",
      "TransportMode": "reliable"
    }
}


channel_name = gen_channel_name(2408, rank)

if rank == 0:
    logging.info("==================I am test_reader_multichannel===================")
logging.info(f"rank {rank:d} / size {size:d}. Channel_name = {channel_name}")

r = reader_gen(cfg_transport, channel_name)
r.Open()

stats = stream_stats()
tstep = 0
while True:
    stepStatus = r.BeginStep(timeoutSeconds=5.0)
    if stepStatus:
        tic = time.time()
        stream_data = r.Get("dummy", save=False)
        logging.info(f"rank {rank:d}, tstep =  {tstep}, mean = {stream_data.mean()}")
        r.EndStep()
        toc = time.time()
        rx_bytes = stream_data.size * 8
        stats.add_transfer(rx_bytes, toc-tic)

        tstep += 1
    else:
        break


logging.info(stats.get_transfer_stats(), stats.get_duration_stats())


# End of file test_reader_multichannel.py