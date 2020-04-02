#-*- coding: UTF-8 -*-

from mpi4py import MPI 
import adios2
import numpy as np 


class reader_base():
    def __init__(self, shotnr, id):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.shotnr = shotnr
        self.id = id
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.IO = self.adios.DeclareIO("stream_{0:03d}".format(self.rank))
        print("reader_base.__init__(): rank = {0:02d}".format(self.rank))


    def Open(self):
        """Opens a new channel"""
        self.channel_name = gen_channel_name(self.shotnr, self.id, 0)
        if self.reader is None:
            self.reader = self.IO.Open(self.channel_name, adios2.Mode.Read)


    def get_data(self, varname):
        """Attempt to load `varname` from the opened stream"""

        var = self.IO.InquireVariable(varname)
        io_array = np.zeros(np.prod(var.Shape()), dtype=np.float)
        self.reader.Get(var, io_array, adios2.Mode.Sync)
        return(io_array)


    def InquireVariable(self, varname):
        """Wrapper for IO.InquireVariable"""
        return self.IO.InquireVariable(varname)


    def BeginStep(self):
        """wrapper for reader.BeginStep()"""
        return self.reader.BeginStep()


    def CurrentStep(self):
        """Wrapper for IO.CurrentStep()"""
        return self.reader.CurrentStep()

    
    def EndStep(self):
        """Wrapper for reader.EndStep()"""
        self.reader.EndStep()


class reader_dataman(reader_base):
    def __init__(self, shotnr, id):
        super().__init__(shotnr, id)
        self.IO.SetEngine("DataMan")
        self.reader = None

        dataman_port = 12300 + self.rank
        transport_params = {"IPAddress": "127.0.0.1",
                            "Port": "{0:5d}".format(dataman_port),
                            "OpenTimeoutSecs": "600",
                            "Verbose": "20"}
        self.IO.SetParameters(transport_params)


class reader_bpfile(reader_base):
        def __init__(self, shotnr, id):
            super().__init__(shotnr, id)
            self.IO.SetEngine("BPFile")
            self.reader = None

# end of file readers.py