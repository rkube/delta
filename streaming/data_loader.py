#-*- Coding: UTF-8 -*-

import h5py
from mpi4py import MPI

from analysis.channels import channel_range

class data_loader():
    """Loads data time-chunk wise for a specified channel range"""


    def __init__(self, filename: str, ch_range: channel_range, chunk_size:int):
        """
        Inputs:
        =======
        filename      : Absolute path + filename to the HDF5 where the shot data is stored
        ch_range      : Range of channels that is processed
        chunk_size    : Number of data elements to load in each iteration

        F.ex.

        >>> ch_rg = channel_range.from_str("L0101-2408")
        >>> loader = data_loader(/data/myhdf5.py, ch_rg, 10_000)

        >>> for _ in range(10):
        >>>    data = loader.get()
        >>>    data is now a list of arrays:
        >>> len(data) = len(ch_rg) # In our case 192
        >>> data[0] = chunk_size

        """
        
        self.ch_range = ch_range
        # Create a list of paths in the HDF5 file, corresponding to the specified channels
        self.filename = filename
        self.chunk_size = chunk_size
        # The current time-chunk. Increases by 1  for each call of get.
        self.current_chunk = 0


    def get(self):
        """Loads the next time-chunk from the data files.

        Returns:
        ========
        data_list: List of data elements from each channel
        """
        data_list = []
        #comm = MPI.COMM_WORLD
        #rank, size = comm.Get_rank(), comm.Get_size()
        #print(f"rank: {rank:d} chunk: {self.current_chunk:d}, channel list: {self.channel_range}")

        with h5py.File(self.filename, "r") as df:
            for ch in [f"/ECEI/ECEI_{c}/Voltage" for c in self.ch_range]:
                data_list.append(df[ch][self.current_chunk * self.chunk_size:
                                        (self.current_chunk + 1) * self.chunk_size])
            self.current_chunk += 1
            df.close()

        return(data_list)


# End of file 