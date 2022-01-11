# -*- Encoding: UTF-8 -*-

import numpy as np

class stream_stats():
    """Collects statistics for data transfer timings."""

    def __init__(self):
        """Initialize the object."""
        # List that stores the size of transferred chunks, in bytes
        self.chunk_sizes = []
        # List that stores the time spent in send calls, in seconds
        self.durations = []
        # List that stores the transfer speeds, in bytes per second
        self.speeds = []
        # Number of added steps
        self.nsteps = 0
        # Optional index
        self.idx_list = []

    def add_transfer(self, num_bytes, duration, idx=None):
        """Adds a new transfer.

        Args:
            num_bytes (int):
                Number of bytes that have been transferred
            duration (float):
                Duration for the transfer, in seconds

        Returns:
            None
        """

        self.chunk_sizes.append(num_bytes / 1024 / 1024)
        self.durations.append(duration)
        self.speeds.append(self.chunk_sizes[-1] / duration)
        if idx is not None:
            self.idx_list.append(idx)

        self.nsteps += 1

    def get_transfer_stats(self):
        """Return max, min, avg, std of packet sizes."""
        arr = np.array(self.chunk_sizes)
        return (arr.sum(), arr.max(), arr.min(), arr.mean(), arr.std())

    def get_duration_stats(self):
        """Return max, min, avg, std of transfer durations."""
        arr = np.array(self.durations)
        return (arr.sum(), arr.max(), arr.min(), arr.mean(), arr.std())

    def get_speed_stats(self):
        """Return max, min, avg, std of transfer durations."""
        arr = np.array(self.speeds)
        return (arr.sum(), arr.max(), arr.min(), arr.mean(), arr.std())


# End of file stream_stats.py