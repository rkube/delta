#-*- coding: UTF-8 -*-

from mpi4py import MPI 
import adios2
import numpy as np 


class reader_base():
    def __init__(self, shotnr, channel):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.shotnr = shotnr
        self.channel = channel
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.IO = self.adios.DeclareIO("stream_{0:03d}".format(self.rank))
        print("reader_base.__init__(): rank = {0:02d}".format(self.rank))


    def Open(self):
        """Opens a new channel"""
        self.channel_name = "{0:05d}_ch{1:d}.bp".format(self.shotnr, self.channel)

        if self.reader is None:
            self.reader = self.IO.Open(self.channel_name, adios2.Mode.Read)


    def BeginStep(self):
        """wrapper for reader.BeginStep()"""
        return self.reader.BeginStep()


    def InquireVariable(self, varname):
        """Wrapper for IO.InquireVariable"""
        return self.IO.InquireVariable(varname)


    def get_data(self, varname):
        """Attempt to load `varname` from the opened stream"""

        var = self.IO.InquireVariable(varname)
        io_array = np.zeros(np.prod(var.Shape()), dtype=np.float)
        self.reader.Get(var, io_array, adios2.Mode.Sync)

        return(io_array)


    def CurrentStep(self):
        """Wrapper for IO.CurrentStep()"""
        return self.reader.CurrentStep()

    
    def EndStep(self):
        """Wrapper for reader.EndStep()"""
        self.reader.EndStep()


class reader_dataman(reader_base):
    def __init__(self, shotnr, channel):
        super().__init__(shotnr, channel)
        self.IO.SetEngine("DataMan")
        self.reader = None

        dataman_port = 12300 + self.rank
        transport_params = {"IPAddress": "127.0.0.1",
                            "Port": "{0:5d}".format(dataman_port),
                            "Timeout": "30",
                            "Verbose": "20"}
        self.IO.SetParameters(transport_params)


class reader_bpfile(reader_base):
        def __init__(self, shotnr, channel):
            super().__init__(shotnr, channel)
            self.IO.SetEngine("BPFile")
            self.reader = None




# end of file readers.py