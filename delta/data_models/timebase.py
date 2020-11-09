# -*- Encoding: UTF-8 -*-

"""Defines time-base classes to be used in streaming settings."""


import numpy as np


class timebase_streaming():
    """Defines a timebase for a data chunk in the stream."""

    def __init__(self, t_start: float, t_end: float, f_sample: float,
                 samples_per_chunk: int, chunk_idx: int):
        """Defines a timebase for a data chunk in the stream.

        Args:
            t_start (float):
                Start time of the data stream, in seconds
            t_end (float):
                End time of the data stream, in seconds
            f_sample (float):
                Sampling frequency, in Hz
            samples_per_chunk (int):
                Number of samples per chunk
            chunk_idx (int):
                Index of the chunk that this timebase is used used

        Returns:
            None
        """
        assert(t_start < t_end)
        assert(f_sample > 0)
        assert(chunk_idx >= 0)
        assert(chunk_idx < (t_end - t_start) * f_sample // samples_per_chunk)

        self.t_start = t_start
        self.t_end = t_end
        self.f_sample = f_sample
        self.dt = 1. / self.f_sample

        # Total samples in the entire stream
        self.total_num_samples = int((self.t_end - self.t_start) / self.dt)
        self.chunk_idx = chunk_idx
        # How many samples are in a chunk
        self.samples_per_chunk = samples_per_chunk

    def time_to_idx(self, time: float):
        """Generates an index suitable to index the current data chunk in time.

        Args:
            time (float):
                Absolute time we wish to get an index for

        Returns:
            tidx_rel (int):
                Relative index in the current time chunk
        """
        assert(time >= self.t_start)
        assert(time <= self.t_end)

        # Generate the index the time would have in the entire time-series
        tidx_absolute = round((time - self.t_start) / self.dt)
        if tidx_absolute // self.samples_per_chunk != self.chunk_idx:
            return None
        tidx_rel = tidx_absolute % self.samples_per_chunk

        return tidx_rel

    def gen_full_timebase(self):
        """Generates an array of times associated with the samples in the current chunk."""
        return np.arange(self.chunk_idx * self.samples_per_chunk,
                         (self.chunk_idx + 1) * self.samples_per_chunk) * self.dt + self.t_start


class timebase():
    """Defines a time base for ECEI channel data."""

    def __init__(self, t_start, t_end, f_sample):
        """Initializes the timebase class.

        Args:
            t_trigger (float):
                Trigger time? in seconds
            t_offset (float):
                No idea, in seconds.
            f_sample (float):
                Sampling frequency of the ECEI data, in Hz.

        Returns:
            None
        """
        # Assume that 0 <= t_start < t_end
        assert(t_end >= 0.0)
        assert(t_start < t_end)
        assert(f_sample >= 0.0)
        self.t_start = t_start
        self.t_end = t_end
        self.f_sample = f_sample
        self.dt = 1. / self.f_sample

        self.num_samples = int((self.t_end - self.t_start) / self.dt)

    def time_to_idx(self, t0):
        """Given a timestamp, returns the index where the timebase is closest.

        Args:
            t0 (float):
                Time stamps that we want to calculate the index for

        Returns:
            idx (int):
                Index for timestamp.
        """
        # Ensure that the time we are looking for is inside the domain
        assert(t0 >= self.t_start)
        assert(t0 <= self.t_end)

        fulltime = self.get_fulltime()
        idx = np.argmin(np.abs(fulltime - t0))

        return idx

    def get_fulltime(self):
        """Returns an array with the full timebase.

        Args:
            None

        Returns:
            trange (ndarray):
                np.arange spanning start and end time using sample spacing dt.

        """
        return np.arange(self.t_start, self.t_end, self.dt)

# End of file timebases.py
