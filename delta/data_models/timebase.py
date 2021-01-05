# -*- Encoding: UTF-8 -*-

"""Defines time-base classes to be used in streaming settings."""


class timebase_streaming():
    """Defines a timebase for a time-chunk of data in a stream.

    The start and end time refer to the entire data stream. To describe the
    position of the time-chunk in the stream, the member function 
    :py:func:`analysis.timebase.timebase_streaming.time_to_idx` returns an
    integer index if the time argument falls within the current time chunk.

    .. code-block:: python
        
        >>> t_start = -0.1   # This is the start time of the stream
        >>> t_end = 9.9  # This is the stop time of the stream
        >>> f_sample = 5e5  # Sampling frequency
        >>> chunk_size = 10_000  # Number of samples per time-chunk
        >>> chunk_idx = 12  # We want to describe the 12th time-chunk.

        >>> tb_stream = timebase_streaming(t_start, t_end, f_sample, chunk_size, chunk_idx)

        >>> for t, target in range(chunk_idx * chunk_size - 2,
                                   chunk_idx * chunk_size + 2)):
                time = tb_stream.t_start + t / tb_stream.f_sample
                print(tb_stream.time_to_idx(time))

            [None, None, 0, 1, 2]
    
    The first two sample times fall before the described time-chunk. Here time_to_idx returns `None`.
    The last three sample times are within time-chunk and time_to_idx returns the time index of 
    the relevant sample.

    """

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

    def get_trange(self):
        """Returns start and end time in this chunk.

        Returns:
            (t0, t1): (tuple[int]):
                Start and end time in this chunk.
        """
        t0 = self.t_start + self.dt * (self.chunk_idx * self.samples_per_chunk)
        t1 = self.t_start + self.dt * ((self.chunk_idx + 1) * self.samples_per_chunk - 1)
        return (t0, t1)

    def time_to_idx(self, time: float):
        """Generates an index suitable to index the current data chunk in time.

        Args:
            time (float):
                Absolute time we wish to get an index for

        Returns:
            tidx_rel (int):
                Relative index in the current time chunk. Returns :code:`None` of time is outside 
                of current chunk.
        """
        assert(time >= self.t_start)
        assert(time <= self.t_end)

        # Generate the index the time would have in the entire time-series
        tidx_absolute = round((time - self.t_start) / self.dt)
        if tidx_absolute // self.samples_per_chunk != self.chunk_idx:
            return None
        tidx_rel = tidx_absolute % self.samples_per_chunk

        return tidx_rel

    # def gen_full_timebase(self):
    #     """Generates an array of times associated with the samples in the current chunk."""
    #     return np.arange(self.chunk_idx * self.samples_per_chunk,
    #                      (self.chunk_idx + 1) * self.samples_per_chunk) * self.dt + self.t_start

    def __str__(self):
        """Pretty print."""
        t0, t1 = self.get_trange()
        print_str = "Class timebase_streaming "
        print_str += f"t_start={self.t_start:6.4f}s "
        print_str += f"t_start={self.t_start:6.4f}s "
        print_str += f"dt={self.dt:6.4f}s "
        print_str += f"chunk_idx={self.chunk_idx}"
        print_str += f" local t0={t0:6.4f}s, t1={t1:6.4f}s"
        return print_str

# End of file timebases.py
