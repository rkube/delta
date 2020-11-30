#-*- Coding: UTF-8 -*-

from mpi4py import MPI

import sys 
import logging
import json
from os.path import join

import numpy as np

#from analysis.channels import channel, channel_range
from streaming.adios_helpers import gen_io_name

import adios2


class reader_base():
    def __init__(self, cfg: dict, stream_name):
        """Generates a reader for KSTAR ECEI data.

        Parameters:
        -----------
        cfg: delta config dictionary
        """

        comm  = MPI.COMM_SELF
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        # This should be MPI.COMM_SELF, not MPI.COMM_WORLD
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.logger = logging.getLogger("simple")

        self.IO = self.adios.DeclareIO(gen_io_name(self.rank))
        # Keeps track of the past chunk sizes. This allows to construct a dummy time base
        self.reader = None
        self.stream_name = stream_name

    def Open(self, multi_channel_id=None):
        """Opens a new channel.

        multi_channel_id (None or int): add suffix for multi-channel
        """
        # We add a suffix for multi-channel
        if multi_channel_id is not None:
            self.channel_name = "%s_%02d"%(self.channel_name, multi_channel_id)

        self.logger.info(f"Waiting to receive channel name {self.stream_name}")
        if self.reader is None:
            self.reader = self.IO.Open(self.stream_name, adios2.Mode.Read)
            #attrs = self.IO.InquireAttribute("cfg")
        else:
            pass
        self.logger.info(f"Opened channel {self.stream_name}")
        #self.logger.info(f"-> attrs = {attrs.DataString()}")

        return None

    def BeginStep(self):
        """Wrapper for reader.BeginStep."""
        res = self.reader.BeginStep()
        if res == adios2.StepStatus.OK:
            return(True)
        return(False)

    def CurrentStep(self):
        """Wrapper for IO.CurrentStep."""
        res = self.reader.CurrentStep()
        return(res)

    def EndStep(self):
        """Wrapper for reader.EndStep."""
        res = self.reader.EndStep()
        return(res)

    def InquireVariable(self, varname: str):
        """Wrapper for IO.InquireVariable."""
        res = self.IO.InquireVariable(varname)
        return(res)

    def InquireAttribute(self, attrname: str):
        """Wrapper for IO.InquireAttribute."""
        res = self.IO.InquireAttribute(attrname)
        return(res)

    def get_data(self, varname: str):
        """Attempts to load `varname` from the opened stream."""
        var = self.IO.InquireVariable(varname)
        if var.Type() == 'int64_t':
            dtype = np.dtype('int64')
        elif var.Type() == 'double':
            dtype = np.dtype('double')
        else:
            dtype = np.dtype(var.Type())
        io_array = np.zeros(var.Shape(), dtype=dtype)
        self.reader.Get(var, io_array, adios2.Mode.Sync)

        return(io_array)

    def get_attrs(self, attrsname: str):
        """Inquire json string `attrsname` from the opened stream.

        Information about the diagnostic configuration is stored as a json
        string in the ADIOS strem. Inquire the string attribute from the stream
        and generate a dictionary from its json interpretation.

        Args:
            attrsname (str):
                Name of the attribute string in the ADIOS channel

        Returns:
            all_cfg["diagnostics"]["parameters"] (dict):
                Named section of the all_cfg
        """
        try:
            attrs = self.IO.InquireAttribute(attrsname)
            stream_attrs = json.loads(attrs.DataString()[0])
        except ValueError as e:
            self.logger.error(f"Could not load attributes from stream: {e}")
            raise ValueError(f"Failed to load attributes {attrsname} from {self.stream_name}")

        self.logger.info(f"Loaded attributes: {stream_attrs}")
        # TODO: Clean up naming conventions for stream attributes
        return stream_attrs

    def Get(self, varname: str, save: bool=False):
        """Get data from varname at current step. This is diagnostic-independent code.

        Args:
            varname (str):
                variable name to inquire from adios stream
            save (bool):
                saves data to numpy if true. Default: False

        Returns:
            time_chunk (ndarray)
                Contains data of the current step
        """
        # elif isinstance(channels, type(None)):
        self.logger.info(f"Reading varname {varname}. Step no. {self.CurrentStep():d}")
        var = self.IO.InquireVariable(varname)
        # self.logger.info(f"Received {varname}, shape={var.Shape()}, ", type(var.Type()))
        if var.Type() == 'double':
            new_dtype = np.float64
        elif var.Type() == 'float':
            new_dtype = np.float32
        else:
            raise ValueError(var.Type())
        time_chunk = np.zeros(var.Shape(), dtype=new_dtype)
        self.reader.Get(var, time_chunk, adios2.Mode.Sync)
        self.logger.info("Got data")

        if save:
            np.savez(f"test_data/time_chunk_tr_s{self.CurrentStep():04d}.npz", time_chunk=time_chunk)

        return time_chunk


class reader_gen(reader_base):
    def __init__(self, cfg: dict, stream_name: str):
        """Instantiates a reader.

           Control Adios method and params through cfg

        Args:
            cfg (dict):
                delta config dict
            stream_name (str):
                Name of the data stream to read
        """
        super().__init__(cfg, stream_name)
        self.IO.SetEngine(cfg["engine"])
        self.IO.SetParameters(cfg["params"])
        self.reader = None

# End of file reader_mpi.py
