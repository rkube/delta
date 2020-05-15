# coding: UTF-8 -*-

import numpy as np
import logging
import timeit

import more_itertools

from analysis.channels import channel, channel_range, channel_pair, unique_everseen
from analysis.kernels_spectral import kernel_null, kernel_crosspower, kernel_crossphase, kernel_coherence, kernel_crosscorr, kernel_bicoherence, kernel_skw
from analysis.kernels_spectral_cy import kernel_coherence_64_cy, kernel_crosspower_64_cy, kernel_crossphase_64_cy
from analysis.task_fft import task_fft_scipy

import backends

from scipy.signal import stft

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

    def __init__(self, task_config, fft_config, ecei_config, storage_config):
        """Initialize the object with a fixed channel list, a fixed name of the analysis to be performed
        and a fixed set of parameters for the analysis routine.

        Inputs:
        =======
        task_config: dict, defines parameters of the analysis to be performed
        fft_config: dict, gives parameters of the fourier-transformed data
        ecei_config: dict, information on ecei diagnostic
        """

        self.task_config = task_config
        self.ecei_config = ecei_config
        self.storage_config = storage_config

        self.logger = logging.getLogger("simple")


        # Stores the description of the task. This can be arbitrary
        self.description = task_config["task_description"]
        # Stores the name of the analysis we are going to execute
        self.analysis = task_config["analysis"]

        if self.analysis == "cross_phase":
            self.kernel = kernel_crossphase_64_cy
        elif self.analysis == "cross_power":
            self.kernel = kernel_crosspower_64_cy
        elif self.analysis == "cross_correlation":
            self.kernel = kernel_crosscorr
        elif self.analysis == "coherence":
            self.kernel = kernel_coherence_64_cy
        elif self.analysis == "skw":
            self.kernel = kernel_skw
        elif self.analysis == "bicoherence":
            self.kernel = kernel_bicoherence
        elif self.analysis == "null":
            self.kernel = kernel_null
        else:
            raise NameError(f"Unknown analysis task {self.analysis}")
        

        # Get the configuration from task_fft_scipy, but don't store the object.
        fft_config["fsample"] = ecei_config["SampleRate"] * 1e3
        self.my_fft = task_fft_scipy(10_000, fft_config, normalize=True, detrend=True)
        self.fft_params = self.my_fft.get_fft_params()

        if self.storage_config["backend"] == "numpy":
            self.storage_backend = backends.backend_numpy(self.storage_config)
        elif self.storage_config["backend"] == "mongo":
            self.store_backend = backends.backend_mongodb(self.storage_config)
        elif self.storage_config["backend"] == "null":
            self.storage_backend = backends.backend_null(self.storage_config)

        # Parse the reference and cross channels.
        self.ref_channels = channel_range.from_str(task_config["ref_channels"])
        # These channels serve as the cross-data for the spectral diagnostics
        self.cmp_channels = channel_range.from_str(task_config["cmp_channels"])

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


    def calc_and_store(self, stream_data, ch_it, info_dict):
        """Dispatches a kernel and stores results

        Parameters:
        -----------
        fft_data - ndarray, complex: Contains the fourier-transformed data. 
                            dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)    
        ch_it - List of channels to iterate over
        info_dict  - metadata for the fft_data object
        """

        try:
            # Calculate the cross phase
            result = self.kernel(fft_data, ch_it, self.fft_params)

            #Store result in the DB
            self.store_backend.store_data(result, info_dict)
            logging.info(f"{self.analysis}: _submit tidx {info_dict['tidx']}, chunk {info_dict['channel_batch']}: Finished")

            # Zero out the result once it has been written
            result = None

        except:
            self.logger.info("Unexpected error in calc_and_store:", sys.exc_info()[0])
            raise 

        return None

    def submit(self, executor, data, tidx):
        """Submits a kernel to the executor
        
        Note: When we are submitting member functions on the executioner we are losing the
        ability to store future lists.

        This implementation lacks self.futures_list and the results of e.submit are discarded.
        """
        info_dict_list = [{"analysis_name": self.analysis,
                           "tidx": tidx,
                           "channel_batch": chunk_idx} for chunk_idx in range(self.num_chunks)]
        
        tic_fft = timeit.default_timer()
        res = executor.submit(stft, data, axis=1, fs=self.fft_params["fs"], nperseg=self.fft_params["nfft"],
                              window=self.fft_params["window"], detrend=self.fft_params["detrend"], 
                              noverlap=self.fft_params["noverlap"], padded=False, return_onesided=False, boundary=None)
        fft_data = res.result()
        fft_data = np.fft.fftshift(fft_data[2], axes=1)
        toc_fft = timeit.default_timer()
        self.logger.info(f"FFT took {(toc_fft - tic_fft):6.4f}s")

        _ = [executor.submit(self.calc_and_store, fft_data, ch_it, info_dict) for ch_it, info_dict in zip(self.get_dispatch_sequence(), info_dict_list)]

        return None




# End of file tasks_mpi.py