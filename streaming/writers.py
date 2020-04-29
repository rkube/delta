# -*- coding: UTF-8 -*-

from mpi4py import MPI
from os.path import join
import adios2
import numpy as np
import json
from contextlib import contextmanager

import logging

from streaming.adios_helpers import gen_channel_name_v2, gen_io_name
from analysis.channels import channel_range

class writer_base():
    def __init__(self, cfg: dict):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.logger = logging.getLogger("simple")

        self.shotnr = cfg["shotnr"]
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.IO = self.adios.DeclareIO(gen_io_name(self.rank))
        self.writer = None
        # Adios2 variable that is defined in DefineVariable
        self.variable = None
        # The shape used to define self.variable
        self.shape = None

        # Generate a descriptive channel name
        self.chrg = channel_range.from_str(cfg["transport"]["channel_range"][self.rank])
        self.channel_name = gen_channel_name_v2(self.shotnr, self.chrg.to_str())
        self.logger.info(f"writer_base: channel_name =  {self.channel_name}")


    def DefineVariable(self, data_name:str, data_array:np.ndarray):
        """Wrapper around DefineVariable

        Input:
        ======
        data_name: Name of data
        data_array: array with same shape and data type that will be sent in 
                             all subsequent steps
        """

        self.shape = data_array.shape
        self.variable =  self.IO.DefineVariable(data_name, data_array, 
                                                data_array.shape, # shape
                                                list(np.zeros_like(data_array.shape, dtype=int)),  # start 
                                                data_array.shape, # count
                                                adios2.ConstantDims)

        return(self.variable)


    def DefineAttributes(self, attrsname: str, attrs: dict):
        """Wrapper around DefineAttribute, takes in dictionary and writes each as an Attribute
        NOTE: Currently no ADIOS cmd to use dict, pickle to stringif

        Input:
        ======
        attrs: Dictionary of key,value pairs to be put into attributes

        """
        attrsstr = json.dumps(attrs)
        self.attrs = self.IO.DefineAttribute(attrsname,attrsstr)

    def Open(self):
        """Opens a new channel. 
        """
        if self.writer is None:
            self.writer = self.IO.Open(self.channel_name, adios2.Mode.Write)

        return None

    def BeginStep(self):
        """wrapper for writer.BeginStep()"""
        return self.writer.BeginStep()

    def EndStep(self):
        """wrapper for writer.EndStep()"""
        return self.writer.EndStep()

    def put_data(self, data:np.ndarray):
        """Opens a new stream and send data through it
        Input:
        ======
        data: ndarray. Data to send.
        """

        assert(data.shape == self.shape)

        if self.writer is not None:
            self.logger.info(f"Putting data: name = {self.variable.Name()}, shape = {data.shape}")
            self.writer.Put(self.variable, data, adios2.Mode.Sync)

        return None

    #RMC - I find relying on this gives segfaults in bp files.
    #Better to explicitly close it in the main program
    #def __del__(self):
    #    """Close the IO."""
    #    if self.writer is not None:
    #        self.writer.Close()

    @contextmanager
    def step(self):
        """ This is for using with with keyword """
        self.writer.BeginStep()
        try:
            yield self
        finally:
            self.writer.EndStep()


class writer_dataman(writer_base):
    def __init__(self, cfg):
        """Perform DataMan specific initialization on top of writer_base."""

        super().__init__(cfg)
        self.IO.SetEngine("DataMan")
        cfg["transport"]["params"].update(Port = str(12306 + self.rank))
        self.IO.SetParameters(cfg["transport"]["params"])


class writer_bpfile(writer_base):
    def __init__(self, cfg):
        """Perform BP4 specific initialization on top of writer_base."""
        super().__init__(cfg)
        self.IO.SetEngine("BP4")
        self.channel_name = gen_channel_name_v2(self.shotnr, self.chrg.to_str())

    
class writer_sst(writer_base):
    def __init__(self, cfg):
        """Perform SST specific initialization on top of writer_base."""
        super().__init__(cfg)
        self.IO.SetEngine("SST")
        self.datapath = cfg["transport"]["datapath"]
        self.channel_name = gen_channel_name_v2(self.shotnr, self.chrg.to_str())

        return None

# End of file a2_sender.py