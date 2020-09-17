# -*- Encoding: UTF-8 -*-

import numpy as np

class stream_stats():
    """Collects statistics over data transfer timings"""

    def __init__(self):
        """Initializes"""

        # List that stores the size of transferred packets in bytes
        self.packet_sizes = []
        # List that stores the time spent in send calls, in seconds
        self.durations = []
        # Number of added steps
        self.nsteps = 0

    def add_transfer(self, num_bytes, duration):
        """Adds a new transfer

        Parameters:
        -----------
        num_bytes: int
        duration: float

        Returns:
        --------
        None
        """

        self.packet_sizes.append(num_bytes)
        self.durations.append(duration)
        self.nsteps += 1


    def get_transfer_stats(self):
        """Return max, min, avg, std of packet sizes"""
        arr = np.array(self.packet_sizes)
        return (arr.sum(), arr.max(), arr.min(), arr.mean(), arr.std())

    def get_duration_stats(self):
        """Return max, min, avg, std of transfer durations"""
        arr = np.array(self.durations)
        return (arr.sum(), arr.max(), arr.min(), arr.mean(), arr.std())


# End of file stream_stats.py