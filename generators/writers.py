# -*- coding: UTF-8 -*-

from mpi4py import MPI
import adios2
import numpy as np


class writer_base():
    def __init__(self, shotnr, id):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        #print("writer_base.__init__(): rank = {0:02d}".format(self.rank))

        self.shotnr = shotnr
        self.id = id
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.IO = self.adios.DeclareIO("stream_{0:03d}".format(self.rank))
        self.writer = None


    def DefineVariable(self, data_array):
        """Wrapper around DefineVariable

        Input:
        ======
        data_array, ndarray: numpy array with sme number of elements and data type that will be sent in 
                             all subsequent steps
        """
        self.io_array = self.IO.DefineVariable("floats", data_array, 
                                               data_array.shape, 
                                               list(np.zeros_like(data_array.shape, dtype=int)), 
                                               data_array.shape, 
                                               adios2.ConstantDims)


    def Open(self):
        """Opens a new channel. 
        """

        self.channel_name = "{0:05d}_ch{1:06d}.bp".format(self.shotnr, self.id)

        if self.writer is None:
            self.writer = self.IO.Open(self.channel_name, adios2.Mode.Write)


    def put_data(self, data):
        """Opens a new stream and send data through it
        Input:
        ======
        data: ndarray, float. Data to send)
        """

        if self.writer is not None:
            self.writer.BeginStep()
            self.writer.Put(self.io_array, data, adios2.Mode.Sync)
            self.writer.EndStep()


    def __del__(self):
        """Close the IO."""
        if self.writer is not None:
            self.writer.Close()



class writer_dataman(writer_base):
    def __init__(self, shotnr, id):
        super().__init__(shotnr, id)
        self.IO.SetEngine("DataMan")
        dataman_port = 12300 + self.rank
        transport_params = {"IPAddress": "127.0.0.1",
                            "Port": "{0:5d}".format(dataman_port),
                            "OpenTimeoutSecs": "600",
                            "Verbose": "20"}
        self.IO.SetParameters(transport_params)


class writer_bpfile(writer_base):
    def __init__(self, shotnr, id):
        super().__init__(shotnr, id)
        self.IO.SetEngine("BP4")
        

# End of file a2_sender.py