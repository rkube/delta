# -*- coding: UTF-8 -*-

from mpi4py import MPI

import sys
import numpy as np
import json
import logging
import time

sys.path.append("/global/homes/r/rkube/software/adios2-current/lib/python3.8/site-packages")
import adios2

from streaming.stream_stats import stream_stats
from streaming.adios_helpers import gen_io_name


class writer_base():
    def __init__(self, cfg: dict, stream_name: str):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.logger = logging.getLogger("simple")

        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.IO = self.adios.DeclareIO(gen_io_name(self.rank))
        self.writer = None
        # Adios2 variable that is defined in DefineVariable
        self.variable = None
        # The shape used to define self.variable
        self.shape = None

        # Generate a descriptive channel name
        self.stream_name = stream_name
        self.logger.info(f"writer_base: stream_name =  {self.stream_name}")

        # To generate statistics
        self.stats = stream_stats()


    def DefineVariable(self, var_name:str, shape:tuple, dtype:type):
        """Wrapper around DefineVariable

        Input:
        ======
        var_name: Variable name assigned to the data
        shape: tuple of ints, shape of the variable
        dtype: datatype of the variable
        """

        self.shape = shape
        self.var_name = var_name
        self.variable = self.IO.DefineVariable(var_name, np.zeros(shape, dtype),
                                               shape, len(shape) * [0], shape, adios2.ConstantDims)
        return(self.variable)


    def DefineAttributes(self, attrsname: str, attrs: dict):
        """Wrapper around DefineAttribute, takes in dictionary and writes each as an Attribute
        NOTE: Currently no ADIOS cmd to use dict, pickle to string

        Input:
        ======
        attrs: Dictionary of key,value pairs to be put into attributes

        """
        attrsstr = json.dumps(attrs)
        self.attrs = self.IO.DefineAttribute(attrsname,attrsstr)

    def Open(self):
        """Opens a new channel. """

        if self.writer is None:
            self.writer = self.IO.Open(self.stream_name, adios2.Mode.Write)

        return None

    def BeginStep(self):
        """wrapper for writer.BeginStep()"""
        return self.writer.BeginStep()

    def EndStep(self):
        """wrapper for writer.EndStep()"""
        return self.writer.EndStep()

    def put_data(self, data_class, attrs: dict):
        """Opens a new stream and send data through it
        Input:
        ======
        data: ndarray. Data to send.
        attrs: dictionary: Additional meta-data
        """

        assert(data_class.data().shape == self.shape)

        if self.writer is not None:
            assert(data_class.data().flags.contiguous)
            # if not data_class.data().flags.contiguous:
            #     data = np.array(data_class.data(), copy=True)
            #     self.writer.Put(self.variable, data, adios2.Mode.Sync)
            # else:
            tic = time.perf_counter()
            self.writer.Put(self.variable, data_class.data(), adios2.Mode.Sync)
            toc = time.perf_counter()

            num_bytes = np.product(data_class.data().shape) * data_class.data().itemsize
            dt = toc - tic
            self.stats.add_transfer(num_bytes, dt)

        return None


    def transfer_stats(self):
        """Returns statistics from the transfer"""

        tr_sum, tr_max, tr_min, tr_mean, tr_std = self.stats.get_transfer_stats()
        du_sum, du_max, du_min, du_mean, du_std = self.stats.get_duration_stats()

        stats_str =  f"Summary:\n"
        stats_str += f"========"
        stats_str += f""
        stats_str += f"    total steps:         {self.stats.nsteps}"
        stats_str += f"    total data (MB):     {(tr_sum / 1024 / 1024)}"
        stats_str += f"    transfer times(sec): {(du_sum)}"
        stats_str += f"    throughput (MB/sec): {tr_sum / 1024 / 1024 / du_sum}"

        return stats_str


class writer_gen(writer_base):
    def __init__(self, cfg, stream_name):
        """Instantiates a writer. Control Adios method and params through
        transport section cfg

        Parameters:
        -----------
        cfg..........: delta config dict. This corresponds to the transport section.
        stream_name..: string, name for the adios data stream
        """

        super().__init__(cfg, stream_name)
        self.IO.SetEngine(cfg["engine"])
        self.IO.SetParameters(cfg["params"])

        if cfg["engine"].lower() == "dataman":
            cfg["params"].update(Port = str(int(cfg["params"]["Port"]) + self.rank))


# End of file writers.py

