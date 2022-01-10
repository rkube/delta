#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""Uses writers in multi-channel setting.

Command

```
$ srun -n NN test_writer_mulitchannel.py
```

"""

from mpi4py import MPI
import adios2
import numpy as np

import logging

import sys
sys.path.append("/global/homes/r/rkube/repos/delta/delta")
from streaming.writers import writer_gen
from streaming.adios_helpers import gen_channel_name
from data_models.base_models import twod_chunk

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logging.basicConfig(level=20,
                    format=f"rank {rank:1d} - " + " %(asctime)15s - %(levelname)s - %(message)s")

# Define config section for transport
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
logging.info(f"Channel_name = {channel_name}")


w = writer_gen(cfg_transport, channel_name)
w.DefineVariable("dummy", (192, 10_000), np.float64)
w.Open()
w.DefineAttributes("strem_attrs", {"test": "yes"})

for tstep in range(1, 100):

    if rank == tstep % size:
        data = np.random.normal(1000.0 * (rank + 1) + tstep, 0.1, size=(192, 10_000))
        chunk = twod_chunk(data)

        w.BeginStep()
        w.put_data(chunk)
        w.EndStep()
        logging.info(f"tstep={tstep}")

w.writer.Close()
