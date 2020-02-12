# coding: UTF-8 -*-

import numpy as np
import logging

import more_itertools

from analysis.channels import channel, channel_range, channel_pair, unique_everseen
from analysis.kernels_spectral import kernel_null, kernel_crosspower, kernel_crossphase, kernel_coherence, kernel_crosscorr, kernel_bicoherence, kernel_skw

from analysis.kernels_spectral_cy import kernel_coherence_cy, kernel_crosspower_cy, kernel_crossphase_cy

import backends


"""
Author: Ralph Kube
This file contains the task_spectral class and its derived classes. Each one implements
an analysis routine from the fluctana package.

The parent class, task_spectral, and implements common methods to all tasks:
* It handles the range of channels for which any analysis will be performed
* It defines how this channel range is divided into sub-ranges
* It defines a storage scheme for results of an analysis
* It defines a storage scheme for the meta data

Each child class defines a unique analysis routine in the calculate method
that is applied to a data chunk. For this it uses a poolexecuter model.
The executor client is called with the tasks analysis method, the data chunk
and a channel range. The results of this calculation are accessible through its
future_list.
"""



class task_spectral():
    """Serves as the super-class for analysis methods"""

    def __init__(self, task_config, fft_config, ecei_config):
        """Initialize the object with a fixed channel list, a fixed name of the analysis to be performed
        and a fixed set of parameters for the analysis routine.

        Inputs:
        =======
        task_config: dict, defines parameters of the analysis to be performed
        fft_config: dict, gives parameters of the fourier-transformed data
        ecei_config: dict, information on ecei diagnostic
        """


        # Stores the description of the task. This can be arbitrary
        self.description = task_config["task_description"]
        # Stores the name of the analysis we are going to execute
        self.analysis = task_config["analysis"]

        if self.analysis == "cross_phase":
            self.kernel = kernel_crossphase_cy
        elif self.analysis == "cross_power":
            self.kernel = kernel_crosspower_cy
        elif self.analysis == "cross_correlation":
            self.kernel = kernel_crosscorr
        elif self.analysis == "coherence":
            self.kernel = kernel_coherence_cy
        elif self.analysis == "skw":
            self.kernel = kernel_skw
        elif self.analysis == "bicoherence":
            self.kernel = kernel_bicoherence
        elif self.analysis == "null":
            self.kernel = kernel_null
        else:
            raise NameError(f"Unknown analysis task {self.analysis}")
        
        # Parse the reference and cross channels.
        self.ref_channels = channel_range.from_str(task_config["ref_channels"])
        # These channels serve as the cross-data for the spectral diagnostics
        self.cmp_channels = channel_range.from_str(task_config["cmp_channels"])

        self.task_config = task_config
        self.fft_config = fft_config
        self.ecei_config = ecei_config

        #self.futures_list = []

        # Construct a list of unique channels
        # F.ex. we have ref_channels [(1,1), (1,2), (1,3)] and cmp_channels = [(1,1), (1,2)]
        # The unique list of channels is then
        # (1,1) x (1,1), (1,1) x (1,2)
        # (1,2) x (1,2) !!! Omits (1,2) x (1,1)
        # (1,3) x (1,1)
        # (1,3) x (1,2)
        channel_pairs = [channel_pair(cr, cx) for cr in self.ref_channels for cx in self.cmp_channels]
        # Make a list, so that we don't exhaust the iterator after the first call.
        self.unique_channels = [i[0] for i in more_itertools.distinct_combinations(channel_pairs, 1)]
        # Number of channel pairs per future
        self.channel_chunk_size = task_config["channel_chunk_size"]
        # Total number of chunks, i.e. the number of futures appended to the list per call to calculate
        self.num_chunks = (len(self.unique_channels) + self.channel_chunk_size - 1) // self.channel_chunk_size


    def get_dispatch_sequence(self, niter=None):
        """Returns an a list of iterables that together span all unique
        combinations of ref_ch x cmp_ch.

        Parameters:
        ===========
        niter, int: Length of the sub-lists we split the list of channel pairs into.
        """

        if niter is None:
            niter = self.channel_chunk_size

        all_chunks = more_itertools.chunked(self.unique_channels, niter)
        return(all_chunks)


    def calc_and_store(self, fft_data, ch_it, fft_config, cfg, info_dict):
        """Dispatches a kernel and stores results

        Parameters:
        -----------
        fft_data - ndarray, complex: Contains the fourier-transformed data. 
                            dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)    
        ch_it - List of channels to iterate over
        fft_config - Calculate properties of the used fft
        cfg        - configuration object
        info_dict  - metadata for the fft_data object
        """


        import sys

        try:

            # Instantiate the backend. Do not do this in the object constructor
            # but here in the function dispatched on the executor.
            if cfg['storage']['backend'] == "numpy":
                store_backend = backends.backend_numpy(cfg['storage'])
            elif cfg['storage']['backend'] == "mongo":
                store_backend = backends.backend_mongodb(cfg)    
            elif cfg['storage']['backend'] == "null":
                store_backend = backends.backend_null(cfg['storage'])

            # Calculate the cross phase
            #logging.info(f"coherence tidx {info_dict['tidx']} chunk{info_dict['channel_batch']}: Starting")
            result = self.kernel(fft_data, ch_it, fft_config)
            #logging.info(f"coherence tidx {info_dict['tidx']} chunk{info_dict['channel_batch']}: Finished calculation")

            #Store result in the DB
            store_backend.store_data(result, info_dict)
            logging.info(f"{self.analysis}: _submit tidx {info_dict['tidx']}, chunk {info_dict['channel_batch']}: Finished")

            # Zero out the result once it has been written
            result = None

        except:
            logging.info("Unexpected error in coherence_store:", sys.exc_info()[0])
            raise 

        return None

    def dummy(self, tidx):
        print("__dummy__")

    
    def submit(self, executor, fft_data, tidx, cfg):
        """Submits a kernel to the executor
        
        Note: When we are submitting member functions on the executioner we are losing the
        ability to store future lists.

        This implementation lacks self.futures_list and the results of e.submit are discarded.
        """
        info_dict_list = [{"analysis_name": self.analysis,
                           "tidx": tidx,
                           "channel_batch": chunk_idx} for chunk_idx in range(self.num_chunks)]
        _ = [executor.submit(self.calc_and_store, fft_data, ch_it, self.fft_config, cfg, info_dict) for ch_it, info_dict in zip(self.get_dispatch_sequence(), info_dict_list)]

        return None


# End of file tasks_mpi.py