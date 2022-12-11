# -*- Encoding: UTF-8 -*-

"""Plots pre-processed data."""

import logging
from os.path import join

from preprocess.plot_ecei import plot_ecei_timeslice

import ray 

@ray.remote
class pre_plot():
    """Plots the pre-processed data and stores it to a file.
    
    Plots are made using the instantaneous data in the pipeline. That is,
    the position of the plot routine in the preprocessing pipeline is
    important.
    """

    def __init__(self, params_pre):
        """Instantiates the pre_plot class as a callable.

        Args:
            params_pre (dictionary):
                Preprocessing section of Delta configuration

        Returns:
            None
        """
        self.time_range = params_pre["time_range"]
        self.plot_dir = params_pre["plot_dir"]
        self.logger = logging.getLogger("simple")

    def process(self, data_chunk, executor=None):
        """Plots the data chunk.

        Args:
            data_chunk (2d image):
                Data chunk to be wavelet transformed.
            executor (PEP-3148-style executor):
                Executor on which to execute.

        Returns:
            data_chunk (2d_image):
                Wavelet-filtered images
        """
        plotter = plot_ecei_timeslice(data_chunk)
        tidx_plot = [data_chunk.tb.time_to_idx(t) for t in self.time_range]

        if tidx_plot[0] is not None:
            self.logger.info(f"Plotting data into {self.plot_dir}. Plotting indices {tidx_plot[0]},\
                {tidx_plot[-1]}")

            # data_chunk.mark_bad_channels(verbose=True)
            for tidx in range(tidx_plot[0], tidx_plot[1]):
                fig = plotter.create_plot(data_chunk, tidx)
                fig.savefig(join(self.plot_dir, f"chunk_{data_chunk.tb.chunk_idx}_{tidx:04d}.png"))

        return data_chunk


# End of file pre_wavelet.py
