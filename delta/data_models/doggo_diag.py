# -*- Encoding: UTF-8 -*-

"""Defines helper function for doggo diagnostic.

This is a dummy diagnostic that servers as a proto-type for other
imaging diagnostics such as KSTAR ECEi or KSTAR IRTV.
"""

# import logging


class doggo_chunk():
    """Class that represents a chunk of dog images.

    This class provides the following interface.

    Creating a time-chunk:

    .. code-block:: python

        chunk = doggo_chunk(images, metadata)

    """
    def __init__(self, data, metadata=None):
        """Creates a doggo_chunk from a given dataset.

        Args:
            data(ndarray, float):
                Image tensor of shape (nimages, width, height, channels)
            metadata:
                Some meta-data.
        """
        self.doggo_data = data
        self.metadata = metadata

    @property
    def data(self):
        """Common interface to data."""
        return self.doggo_data

    @property
    def shape(self):
        """Forwards to self.doggo_data.shape."""
        return self.doggo_data.shape

# End of file doggo_diag.py
