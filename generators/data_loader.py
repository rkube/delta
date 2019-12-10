#-*- Coding: UTF-8 -*-

import h5py
from mpi4py import MPI

class data_loader(object):
    """Loads data from KSTAR hdf5 files, batch-wise"""


    def __init__(self, filename, channel_range, batch_size):
        """
        Inputs:
        =======
        filename      : Path to the HDF5 where the shot data is stored
        channel_range : Range of channels (integers) to process
        batch_size    : Number of data elements to load in each iteration

        """
        
        self.channel_range = channel_range
        # Create a list of paths in the HDF5 file, corresponding to the specified channels
        self.channel_range_hdf5 = ["/ECEI/ECEI_L{0:4d}/Voltage".format(c) for c in self.channel_range]
        self.filename = filename
        self.batch_size = batch_size
        # The current batch. Increases by 1  for each call of get.
        self.current_batch = 0


    def get(self):
        """Loads the next batch from the data files.

        Returns:
        ========
        data_list: List of data elements from each channel
        """
        data_list = []
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()

        print("rank: {0:d} batch: {1:d}".format(rank,  self.current_batch), ", channel list: ", self.channel_range)


        with h5py.File(self.filename, "r") as df:
            for ch in self.channel_range_hdf5:
                print("Accessing channel ", ch)
                data_list.append(df[ch][self.current_batch * self.batch_size:
                                        (self.current_batch + 1) * self.batch_size])
            self.current_batch += 1
            df.close()

        return(data_list)

# End of file 
