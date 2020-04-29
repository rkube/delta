#-*- Coding: UTF-8 -*-

import h5py
from mpi4py import MPI
import numpy as np

from analysis.channels import channel_range

class loader_h5():
    """Loads KSTAR ECEi data time-chunk wise for a specified channel range from an HDF5 file"""


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
        self.num_chunks = 500
        # The current time-chunk. Increases by 1  for each call of get.
        self.current_chunk = 0


    def get(self):
        """Loads the next time-chunk from the data files.

        Returns:
        ========
        data_list: List of data elements from each channel
        """
        data_list = []

        with h5py.File(self.filename, "r") as df:
            for ch in [f"/ECEI/ECEI_{c}/Voltage" for c in self.ch_range]:
                data_list.append(df[ch][self.current_chunk * self.chunk_size:
                                        (self.current_chunk + 1) * self.chunk_size])
            self.current_chunk += 1
            df.close()

        return(data_list)

    def get_batch(self):
        """Returns a list with all time chunks. Loads all channels batch-wise and
           splits using numpy.split"""

        data_arr = np.zeros((self.ch_range.length(), 5000000), dtype=np.int16)
        with h5py.File(self.filename, "r") as df:
            for ch in self.ch_range:
                data_arr[ch.idx(), :] = df[f"/ECEI/ECEI_{ch.__str__()}/Voltage"]

        data_arr = data_arr.astype(np.float64)

        return(np.split(data_arr, self.num_chunks, axis=1))



# End of file 