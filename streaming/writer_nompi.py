# -*- coding: UTF-8 -*-

from os.path import join
import adios2
import numpy as np
import json

import logging

from streaming.adios_helpers import gen_channel_name_v3, gen_io_name
from analysis.channels import channel_range

class writer_base():
    def __init__(self, cfg: dict, shotnr: int=18431):

        self.logger = logging.getLogger("simple")

        self.shotnr = shotnr
        self.adios = adios2.ADIOS()
        self.IO = self.adios.DeclareIO(gen_io_name(0))
        self.writer = None
        # Adios2 variable that is defined in DefineVariable
        self.variable = None
        # The shape used to define self.variable
        self.shape = None

        # Generate a descriptive channel name
        self.chrg = channel_range.from_str(cfg["channel_range"][0])
        self.channel_name = gen_channel_name_v3(cfg["datapath"], self.shotnr, self.chrg.to_str())
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
        NOTE: Currently no ADIOS cmd to use dict, pickle to string

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
        if self.writer is not None:
            return self.writer.BeginStep()

        else:
            raise AssertionError("writer not opened")


    def EndStep(self):
        """wrapper for writer.EndStep()"""
        if self.writer is not None:
            return self.writer.EndStep()

        else:
            raise AssertionError("writer not opened")

    def put_data(self, data:np.ndarray):
        """Opens a new stream and send data through it
        Input:
        ======
        data: ndarray. Data to send.
        """

        assert(data.shape == self.shape)
        

        if self.writer is not None:
            self.logger.debug(f"Putting data: name = {self.variable.Name()}, shape = {data.shape}")
            if not data.flags.contiguous:
                data = np.array(data, copy=True)
            self.writer.Put(self.variable, data, adios2.Mode.Sync)

        return None


class writer_gen(writer_base):
    def __init__(self, cfg, shotnr: int=18431):
        """Instantiates a writer.
           Control Adios method and params through cfg

        Parameters:
        -----------
        cfg : delta config dict
        """
 
        super().__init__(cfg, shotnr)
        self.IO.SetEngine(cfg["engine"])
        self.IO.SetParameters(cfg["params"])

        if cfg["engine"].lower() == "dataman":
            cfg["params"].update(Port = str(int(cfg["params"]["Port"]) + 0))


# End of file writers.pyf