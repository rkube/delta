#-*- coding: UTF-8 -*-

from mpi4py import MPI 
from os.path import join

import adios2
import numpy as np 
import json

import logging

from streaming.adios_helpers import gen_io_name, gen_channel_name_v2
from analysis.channels import channel_range

class reader_base():
    """Base class for MPI based data readers.

    IO name is {shotnr:05d}_ch{channel_id:03d}_r{rank:03d}.bp

    A reader receives time-step data on a channel name based on a shotnr,
    a channel_id and an MPI rank.
    """
    
    def __init__(self, cfg: dict):

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+                                                                                    +")
        print("+      readers.py is deprecated. Use reader_mpi.py or reader_nompi.py                +")
        print("+                                                                                    +")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")        
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.logger = logging.getLogger("simple")

        self.shotnr = cfg["shotnr"]
        self.adios = adios2.ADIOS(MPI.COMM_WORLD)
        self.IO = self.adios.DeclareIO(gen_io_name(self.shotnr))
        self.reader = None

        # Generate a descriptive channel name
        chrg = channel_range.from_str(cfg["transport"]["channel_range"][self.rank])
        self.channel_name = gen_channel_name_v2(self.shotnr, chrg.to_str())
        self.logger.info(f"reader_base: channel_name =  {self.channel_name}")



    def Open(self):
        """Opens a new channel.

        When using BP4, this will open file file self.channel_name.
        When using DataMan, this will wait to connect to the channel and host
        specified in config['transport']['params']. Open will wait for Timeout
        seconds until it throws an error.
        """

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+                                                                                    +")
        print("+      readers.py is deprecated. Use reader_mpi.py or reader_nompi.py                +")
        print("+                                                                                    +")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")      

        self.logger.info(f"Waiting to receive {self.channel_name}")
        if self.reader is None:
            self.reader = self.IO.Open(self.channel_name, adios2.Mode.Read)

        return None


    def BeginStep(self):
        """wrapper for reader.BeginStep()"""
        return self.reader.BeginStep()


    def InquireVariable(self, varname):
        """Wrapper for IO.InquireVariable"""
        return self.IO.InquireVariable(varname)


    def get_data(self, varname):
        """Attempt to load `varname` from the opened stream"""


        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+                                                                                    +")
        print("+      readers.py is deprecated. Use reader_mpi.py or reader_nompi.py                +")
        print("+                                                                                    +")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")      

        var = self.IO.InquireVariable(varname)
        io_array = np.zeros(var.Shape(), dtype=np.float32)
        self.reader.Get(var, io_array, adios2.Mode.Sync)

        return(io_array)


    def get_attrs(self, attrsname):
        """Get json string `attrsname` from the opened stream"""

        attrs = self.IO.InquireAttribute(attrsname)
        return json.loads(attrs.DataString()[0])


    def CurrentStep(self):
        """Wrapper for IO.CurrentStep()"""
        return self.reader.CurrentStep()

    
    def EndStep(self):
        """Wrapper for reader.EndStep()"""
        self.reader.EndStep()


class reader_dataman(reader_base):
    """Reader that uses the DataMan Engine."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.IO.SetEngine("DataMan")
        cfg["transport"]["params"].update(Port = str(12306 + self.rank))

        logging.info(f"reader_dataman: params = {cfg['transport']['params']}")
        self.IO.SetParameters(cfg["transport"]["params"])


class reader_bpfile(reader_base):
    """Reader that uses the BP4 engine."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.IO.SetEngine("BP4")
        self.datapath = "/global/homes/r/rkube/repos/delta"

    def Open(self):
        if self.reader is None:
            self.reader = self.IO.Open("test.bp4", adios2.Mode.Read)


class reader_sst(reader_base):
    """Reader that uses the SST engine."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.IO.SetEngine("SST")
        self.datapath = cfg["transport"]["datapath"]

    def Open(self):
        """Opens a new channel. Expects the SST contact information in self.datapath."""
        
        self.logger.info(f"Waiting to receive {self.channel_name}")
        if self.reader is None:
            self.reader = self.IO.Open(join(self.datapath, self.channel_name), adios2.Mode.Read)

        return None


# class reader_dataman(reader_base):
#     def __init__(self, shotnr, id):
#         super().__init__(shotnr, id)
#         self.IO.SetEngine("DataMan")

#         dataman_port = 12300 + self.rank
#         transport_params = {"IPAddress": "203.230.120.125",
#                             "Port": "{0:5d}".format(dataman_port),
#                             "OpenTimeoutSecs": "600",
#                             "AlwaysProvideLatestTimestep": "true"}
#         self.IO.SetParameters(transport_params)
#         self.logger.info(">>> reader_dataman ... ")


# class reader_bpfile(reader_base):
#     def __init__(self, shotnr, id):
#         super().__init__(shotnr, id)
#         self.IO.SetEngine("BP4")
#         self.IO.SetParameter("OpenTimeoutSecs", "600")
#         self.logger.info(">>> reader_bpfile ... ")

# class reader_sst(reader_base):
#     def __init__(self, shotnr, id):
#         super().__init__(shotnr, id)
#         self.IO.SetEngine("SST")
#         self.IO.SetParameters({"OpenTimeoutSecs": "600.0"})
#         self.logger.info(">>> reader_sst ... ")

# class reader_gen(reader_base):
#     """ General reader to be initialized by name and parameters
#     """
#     def __init__(self, shotnr, id, engine, params):
#         super().__init__(shotnr, id)
#         self.IO.SetEngine(engine)
#         _params = params
#         if engine.lower() == "dataman":
#             dataman_port = 12300 + self.rank
#             _params.update(Port = "{0:5d}".format(dataman_port))
#         self.IO.SetParameters(_params)
#         self.reader = None

# end of file readers.py