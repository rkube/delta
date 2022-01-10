#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

"""Uses writers in multi-channel setting.

Command

```
$ python test_writer_mulitchannel.py
```

"""

from mpi4py import MPI
import adios2
import numpy as np

import sys
sys.path.append("/global/homes/r/rkube/repos/delta/delta")
from streaming.writers import writer_gen
from streaming.adios_helpers import gen_channel_name
from data_models.base_models import twod_chunk

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Define config section for transport
cfg_transport = {
    "datapath": "tmp_nersc",
    "engine": "BP4",
    "params":
    {
      "IPAddress": "128.55.205.18",
      "Timeout": "120",
      "Port": "50001",
      "TransportMode": "fast"
    }
}

channel_name = gen_channel_name(2408, rank)
data = np.random.normal(rank + 1.0, 0.1, size=(192, 10_000))

if rank == 0:
    print("==================I am test_writer_multichannel===================")
print(f"rank {rank:d} / size {size:d}. Channel_name = {channel_name}")

w = writer_gen(cfg_transport, channel_name)
w.DefineVariable("dummy", (192, 10_000), np.float64)
w.Open()
w.DefineAttributes("strem_attrs", {"test": "yes"})

for tstep in range(1, 5):
    data = np.random.normal(100.0 * (rank + 1) + tstep, 0.1, size=(192, 10_000))
    chunk = twod_chunk(data)

    w.BeginStep()
    w.put_data(chunk)
    w.EndStep()

w.writer.Close()
