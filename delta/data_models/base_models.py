#-*- coding: UTF-8 -*-

"""Base classes."""

import numpy as np

class twod_chunk():
    """Base class for two-dimensional data.

    This defines the interface only.
    """
    global data_ 
    #global chunk_data
    
    def __init__(self, data):
        # We should ensure that the data is contiguous so that we can remove this from
        # if not data.flags.contiguous:
        self.data_ =data
        self.chunk_data = np.require(self.data_, dtype= self.data_.dtype, requirements=['C', 'O', 'W', 'A'])
        assert(self.chunk_data.flags.contiguous)

    @property
    def data(self):
        """Common interface to data."""
        return self.chunk_data

    @property
    def shape(self):
        """Forwards to self.chunk_data.shape."""
        return self.chunk_data.shape
    
    def update_data(self,new_data):
        self.chunk_data = np.require(self.data_, dtype=self.data_.dtype, requirements=['C', 'O', 'W', 'A'])
        self.chunk_data[:] = new_data[:]
        
