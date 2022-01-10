#-*- coding: UTF-8 -*-

"""Base classes."""

import numpy as np

class twod_chunk():
    """Base class for two-dimensional data.

    This defines the interface only.
    """

    def __init__(self, data):
        # We should ensure that the data is contiguous so that we can remove this from
        # if not data.flags.contiguous:
        self.chunk_data = np.require(data, dtype=np.float64, requirements=['C', 'O', 'W', 'A'])
        assert(self.chunk_data.flags.contiguous)

    @property
    def data(self):
        """Common interface to data."""
        return self.chunk_data

    @property
    def shape(self):
        """Forwards to self.chunk_data.shape."""
        return self.chunk_data.shape
