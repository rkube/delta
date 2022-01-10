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

import sys
sys.path.append("/global/homes/r/rkube/repos/delta/delta")
from streaming.reader_mpi import reader_gen
from streaming.adios_helpers import gen_channel_name
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
      "IPAddress": "128.55.205.18",
      "Timeout": "120",
      "Port": "50001",
      "TransportMode": "fast"
    }
}


channel_name = gen_channel_name(2408, rank)

if rank == 0:
    logging.info("==================I am test_reader_multichannel===================")
logging.info(f"rank {rank:d} / size {size:d}. Channel_name = {channel_name}")

r = reader_gen(cfg_transport, channel_name)
r.Open()

for tstep in range(1, 100):
    stepStatus = r.BeginStep(timeoutSeconds=5.0)
    if stepStatus:
        stream_data = r.Get("dummy", save=False)
        logging.info(f"rank {rank:d}, tstep = {tstep}, mean = {stream_data.mean()}")
        r.EndStep()


# End of file test_reader_multichannel.py