# -*- coding: UTF-8 -*-

from mpi4py import MPI

import numpy as np
import json
import logging
import time

import adios2

from streaming.stream_stats import stream_stats
from streaming.adios_helpers import gen_io_name


class writer_base():
    """Generc base class for all ADIOS2 writers."""
    def __init__(self, cfg: dict, stream_name: str):
        """Initialize writer_base."""
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

        # To generate statistics
        self.stats = stream_stats()

    def DefineVariable(self, var_name: str, shape: tuple, dtype: type):
        """Wrapper around adios2 DefineVariable method.

        Args:
            var_name (str):
                Variable name assigned to the data
            shape (tuple[int]):
                tuple of ints, shape of the variable
            dtype (type):
                datatype of the variable

        Returns:
            self.variable (adios2.variable)
        """
        self.var_name = var_name
        self.shape = shape
        self.dtype = dtype
        self.variable = self.IO.DefineVariable(var_name, np.zeros(shape, dtype),
                                               shape, len(shape) * [0], shape, adios2.ConstantDims)
        return(self.variable)

    def DefineAttributes(self, attrsname: str, attrs: dict):
        """Wrapper around DefineAttribute.

        Serializes an attribute dictionary as a json, which is written as an attribute
        into ADIOS stream.

        NOTE: Currently no ADIOS cmd to use dict, pickle to string

        Args:
            attrs (dict):
                Dictionary of key,value pairs to be put into attributes

        Returns:
            self.attrs (adios.attributes)

        """
        attrsstr = None
        try:
            attrsstr = json.dumps(attrs)
        except TypeError as e:
            self.logger.error(f"Can't serialize attributes: {e}")
            raise TypeError(f"Can't serialize data stream attributes: {e}")
        self.attrs = self.IO.DefineAttribute(attrsname, attrsstr)
        self.logger.info(f"writer_base: Defined attribute {attrsname}")

    def Open(self):
        """Opens a new channel."""
        if self.writer is None:
            self.writer = self.IO.Open(self.stream_name, adios2.Mode.Write)

    def Close(self):
        """Wrapper for Close."""
        self.writer.Close()

    def BeginStep(self):
        """Wrapper around writer.BeginStep."""
        return self.writer.BeginStep()

    def EndStep(self):
        """Wrapper around writer.EndStep."""
        return self.writer.EndStep()

    def put_data(self, data_class):
        """Opens a new stream and send data through it.

        ADIOS2 requires that the data array is contiguous in memory in
        order to be written.

        Args:
            data_class (2d-array type)
                Data to send.
            attrs (dict):
                Additional meta-data

        Returns:
            None
        """
        assert(data_class.data.shape == self.shape)

        if self.writer is not None:
            # Assert that the data is continuous, as implicitly required by ADIOS2.
            # The burden to produce contiguous data is on the data producer.
            assert(data_class.data.flags.contiguous)
            tic = time.perf_counter()
            self.writer.Put(self.variable, data_class.data, adios2.Mode.Sync)
            toc = time.perf_counter()

            num_bytes = np.product(data_class.data.shape) * data_class.data.itemsize
            dt = toc - tic
            self.stats.add_transfer(num_bytes, dt)

    def transfer_stats(self):
        """Calculates bandwidth statistics from the transfer."""
        tr_sum, tr_max, tr_min, tr_mean, tr_std = self.stats.get_transfer_stats()
        du_sum, du_max, du_min, du_mean, du_std = self.stats.get_duration_stats()

        stats_str = "Summary:\n"
        stats_str += "========"
        stats_str += ""
        stats_str += f"    total steps:         {self.stats.nsteps}"
        stats_str += f"    total data (MB):     {(tr_sum / 1024 / 1024)}"
        stats_str += f"    transfer times(sec): {(du_sum)}"
        stats_str += f"    throughput (MB/sec): {tr_sum / 1024 / 1024 / du_sum}"

        return stats_str


class writer_gen(writer_base):
    """I don't know why this is here - RK 2020-11-29."""
    def __init__(self, cfg_transport, stream_name):
        """Instantiates a writer.

        Control Adios method and params through transport section cfg

        Args:
            cfg_transport (dict):
                This corresponds to the transport section.
            stream_name (str):
                Name for the adios data stream

        Returns:
            None
        """
        super().__init__(cfg_transport, stream_name)
        self.IO.SetEngine(cfg_transport["engine"])
        self.IO.SetParameters(cfg_transport["params"])

        if cfg_transport["engine"].lower() == "dataman":
            cfg_transport["params"].update(Port=str(int(cfg_transport["params"]["Port"]) +
                                           self.rank))


# End of file writers.py

