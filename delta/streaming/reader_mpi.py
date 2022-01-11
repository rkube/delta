#-*- Coding: UTF-8 -*-

from mpi4py import MPI

import logging
import json
import numpy as np
import adios2

from streaming.adios_helpers import gen_io_name
from streaming.stream_stats import stream_stats


class reader_gen():
    def __init__(self, cfg_transport: dict, stream_name: str):
        """Initializes the generic reader base class.

        Args:
            cfg_transport (dict):
                delta config dict
            stream_name (str):
                Name of the data stream to read

        Returns:
            A class instance

        Used keys from cfg:
            * transport.engine - Defines the `ADIOS2 engine <https://adios2.readthedocs.io/en/latest/engines/engines.html#supported-engines>`_
            * transport.params - Passed to `SetParameters <https://adios2.readthedocs.io/en/latest/api_full/api_full.html?highlight=setparameters#_CPPv4N6adios22IO13SetParametersERKNSt6stringE>`_
  
        """
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        # This should be MPI.COMM_SELF, not MPI.COMM_WORLD
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.logger = logging.getLogger("simple")

        self.IO = self.adios.DeclareIO(gen_io_name(self.rank))
        self.IO.SetEngine(cfg_transport["engine"])
        if cfg_transport["engine"].lower() == "dataman":
            cfg_transport["params"].update(Port=str(int(cfg_transport["params"]["Port"]) +
                                           2 * self.rank))

            self.logger.info(f"rank: {self.rank:d} - port = {cfg_transport['params']['Port']}")
        self.IO.SetParameters(cfg_transport["params"])
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
            # attrs = self.IO.InquireAttribute("cfg")
        else:
            pass
        self.logger.info(f"Opened channel {self.stream_name}")
        # self.logger.info(f"-> attrs = {attrs.DataString()}")

        return None

    def BeginStep(self, timeoutSeconds=0.0):
        """Wrapper for reader.BeginStep."""
        res = self.reader.BeginStep(adios2.StepMode.Read, timeoutSeconds=timeoutSeconds)
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
        #try:
        attrs = self.IO.InquireAttribute(attrsname)
        self.logger.info(f"Got attribute: {attrs.Data()}")
        stream_attrs = json.loads(attrs.DataString()[0])
        #except ValueError as e:
        #    self.logger.error(f"Could not load attributes from stream: {e}")
        #    raise ValueError(f"Failed to load attributes {attrsname} from {self.stream_name}")

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
        if var.Type() == 'double':
            new_dtype = np.float64
        elif var.Type() == 'float':
            new_dtype = np.float32
        elif var.Type() == "int32_t":
            new_dtype = np.int32
        else:
            raise ValueError(var.Type())
        time_chunk = np.zeros(var.Shape(), dtype=new_dtype)
        self.reader.Get(var, time_chunk, adios2.Mode.Sync)
        self.logger.info("Got data")

        if save:
            np.savez(f"test_data/time_chunk_tr_s{self.CurrentStep():04d}.npz", time_chunk=time_chunk)

        return time_chunk


# End of file reader_mpi.py
