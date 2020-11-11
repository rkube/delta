# coding: UTF-8 -*-

"""Couple analysis kernels to the MPI launch mechanism.

This file contains methods to launch analysis kernels on the analysis pipeline.

The method calc_and_store is used to wrap kernel execution and store results.
This is used to glue together data, kernel, and parameters.

The class task_spectral launches code on an executor. It passes the correct analysis
kernel, parameters and data on the executor.

The class task_list is a convenience class that stores multiple task_spectral objects.
"""

import logging

import more_itertools

# Import plain python kernels
# from analysis.kernels_spectral import kernel_null, kernel_crosspower
# from analysis.kernels_spectral import kernel_crossphase, kernel_coherence kernel_crosscorr
# from analysis.kernels_spectral import kernel_bicoherence, kernel_skw
# Import cython kernels
from analysis.kernels_spectral import kernel_crosscorr
from analysis.kernels_spectral_cy import kernel_coherence_64_cy
from analysis.kernels_spectral_cy import kernel_crosspower_64_cy, kernel_crossphase_64_cy
# Import CUDA kernels
# from analysis.kernels_spectral_cu import kernel_crossphase_cu, kernel_crosscorr_cu

from data_models.channels_2d import channel_pair
from data_models.helpers import gen_channel_range, unique_everseen

from storage.backend import get_storage_object


# from scipy.signal import stft
# import cupy as cp
# from cusignal.spectral_analysis.spectral import stft


def calc_and_store(kernel, storage_backend, fft_data, ch_it, info_dict):
    """Dispatch a kernel and store the result.

    Args:
        fft_data (ecei_chunk_ft):
            Contains the fourier-transformed data.
            dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
        ch_it (iterable):
            List of channels to iterate over
        info_dict (dict):
            Metadata for the fft_data object

    Returns:
        None
    """
    from mpi4py import MPI
    import datetime
    from socket import gethostname

    comm = MPI.COMM_WORLD
    tidx = info_dict['tidx']
    an_name = info_dict["analysis_name"]
    t1 = datetime.datetime.now()
    result = kernel(fft_data.data, ch_it, fft_data.fft_params)
    t2 = datetime.datetime.now()
    # dt_calc = t2 - t1
    # #
    # t1 = datetime.datetime.now()
    # #storage_backend.store_data(result, info_dict)
    # t2 = datetime.datetime.now()
    dt_io = -1.0   # t2 - t1

    with open(f"outfile_{(comm.rank):03d}.txt", "a") as df:
        df.write(f"rank {comm.rank:03d}/{comm.size:03d}: tidx={tidx} {an_name} start " +
                 t1.isoformat(sep=" ") + " end " + t2.isoformat(sep=" ") +
                 f" Storage: {dt_io} " + f" {gethostname()}\n")
        df.flush()

    return result


class task_spectral():
    """Serves as the super-class for analysis methods."""

    def __init__(self, task_config, cfg_diagnostic, cfg_storage):
        """Initializes a spectral task.

        Fix channel list, analysis type, and parameters for the analysis routine.

        Args:
            task_config (dict):
                Defines parameters of the analysis to be performed
            diag_config (dict):
                Information on ecei diagnostic

        Returns:
            None
        """
        self.task_config = task_config
        self.cfg_diagnostic = cfg_diagnostic
        self.cfg_storage = cfg_storage
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
        # elif self.analysis == "skw":
        #     self.kernel = kernel_skw
        # elif self.analysis == "bicoherence":
        #     self.kernel = kernel_bicoherence
        # elif self.analysis == "null":
        #     self.kernel = kernel_null
        else:
            raise NameError(f"Unknown analysis task {self.analysis}")

        # Parse the reference and cross channels.
        self.ref_channels = gen_channel_range(cfg_diagnostic, task_config["ref_channels"])
        # These channels serve as the cross-data for the spectral diagnostics
        self.cmp_channels = gen_channel_range(cfg_diagnostic, task_config["cmp_channels"])

        # Construct a list of unique channels
        # F.ex. we have ref_channels [(1,1), (1,2), (1,3)] and cmp_channels = [(1,1), (1,2)]
        # The unique list of channels is then
        # (1,1) x (1,1), (1,1) x (1,2)
        # (1,2) x (1,2) !!! Omits (1,2) x (1,1)
        # (1,3) x (1,1)
        # (1,3) x (1,2)
        channel_pairs = [channel_pair(cr, cx)
                         for cr in self.ref_channels for cx in self.cmp_channels]
        # Make a list, so that we don't exhaust the iterator after the first call.
        # self.unique_channels = [i[0] for i in
        #                         more_itertools.distinct_combinations(channel_pairs, 1)]
        self.unique_channels = [ch for ch in unique_everseen(channel_pairs)]
        # Number of channel pairs per future
        self.channel_chunk_size = task_config["channel_chunk_size"]
        # Total number of chunks, i.e. number of futures appended to the list per call to calculate
        self.num_chunks = (len(self.unique_channels) +
                           self.channel_chunk_size - 1) // self.channel_chunk_size

        storage_class = get_storage_object(self.cfg_storage)
        self.storage_backend = storage_class(self.cfg_storage)
        self.storage_backend.store_metadata(self.task_config, self.get_dispatch_sequence())

    def get_dispatch_sequence(self, niter=None):
        """Returns an a list of iterables that span all unique combinations of ref_ch x cmp_ch.

        Args:
            niter (int):
                Length of the sub-lists we split the list of channel pairs into.

        Returns:
            all_chunks (???):
                Chunked dispatch sequence.
        """
        if niter is None:
            niter = self.channel_chunk_size

        all_chunks = more_itertools.chunked(self.unique_channels, niter)
        return(all_chunks)

    def submit(self, executor, fft_data):
        """Launches a spectral analysis kernel on an executor.

        Args:
        executor (PEP-3148-style executor):
            Executor to use
        fft_data (data-model):
            Fourier Coefficients of the data to analyze

        Returns:
            None

        Note: When submitting member functions on the executioner we are losing the
        ability to store future lists.

        This implementation lacks self.futures_list and the results of e.submit are discarded.
        """
        self.logger.info("Entering submit.")

        info_dict_list = [{"analysis_name": self.analysis,
                           "chunk_idx": fft_data.tb.chunk_idx,
                           "channel_batch": batch_idx} for batch_idx in range(self.num_chunks)]

        _ = [executor.submit(calc_and_store,
                             self.kernel,
                             self.storage_backend,
                             fft_data,
                             ch_it,
                             info_dict) for ch_it, info_dict in zip(self.get_dispatch_sequence(),
                                                                    info_dict_list)]
        self.logger.info(f"chunk_idxidx={fft_data.tb.chunk_idx} submitted {self.analysis} " +
                         "as {self.num_chunks} tasks.  fft_data: {type(fft_data)}")

        # for fut, info_dict in zip(fut_list, info_dict_list):
        #     result = fut.result()
        #     tic_io = time.perf_counter()
        #     self.storage_backend.store_data(result, info_dict)
        #     toc_io = time.perf_counter()
        #     dt_io = toc_io - tic_io

        #     size_in_MB = np.prod(result.shape) * result.dtype.itemsize / 1024 / 1024
        #     self.logger.info(f"Storing result for tidx={tidx} {self.analysis}:
        #                      {size_in_MB:6.4f}MB took {dt_io:4.2f}s")

        return None


class task_list():
    """Defines interface to execute a group of tasks on an PEP-3148 executor."""

    def __init__(self, executor_anl, task_config_list, diag_config, cfg_storage):
        """Initialize the object with a list of tasks to be performed.

        These tasks share a common channel list.

        Args:
            executor_anl (PEP-3148 executor):
                Executor on which analysis tasks are performed
            task_list (dict):
                Defines parameters for the analysis routines
            diag_config (dict):
                Diagnostic section of Delta config
            cfg_storage (dict,):
                Storage section of the Delta config

        Returns:
            None
        """
        self.executor_anl = executor_anl
        self.task_config_list = task_config_list
        # Don't store fft_config but use fft_params from one of the tasks instead.
        # Do this since we need the sampling frequency, which is calculated from ECEi data.
        self.diag_config = diag_config
        self.cfg_storage = cfg_storage

        self.logger = logging.getLogger("simple")

        self.task_list = []
        for task_cfg in self.task_config_list:
            self.task_list.append(task_spectral(task_cfg, self.diag_config, self.cfg_storage))

        # self.fft_params = self.task_list[0].fft_params

    def submit(self, data_chunk):
        """Launches the analysis pipeline with a data chunk.

        Args:
            data (data_chunk, data_model):
                A time chunk of data. This is a data_model derived class.

        Returns:
            None
        """
        # tic_fft = time.perf_counter()

        # # # Following 4 lines execute the stft on the GPU
        # data_gpu = cp.asarray(data_chunk.data)
        # res = self.executor_fft.submit(stft,
        #                                data_gpu,
        #                                axis=data_chunk.axis_t,
        #                                fs=self.fft_params["fs"],
        #                                nperseg=self.fft_params["nfft"],
        #                                window=self.fft_params["window"],
        #                                detrend=self.fft_params["detrend"],
        #                                noverlap=self.fft_params["noverlap"],
        #                                padded=False,
        #                                return_onesided=False,
        #                                boundary=None)
        # fft_data_tmp = res.result()
        # fft_data = cp.asnumpy(fft_data_tmp[2])

        # res = self.executor_fft.submit(stft,
        #                                data_chunk.data,
        #                                axis=1,
        #                                fs=self.fft_params["fs"],
        #                                nperseg=self.fft_params["nfft"],
        #                                window=self.fft_params["window"],
        #                                detrend=self.fft_params["detrend"],
        #                                noverlap=self.fft_params["noverlap"],
        #                                padded=False,
        #                                return_onesided=False,
        #                                boundary=None)
        # fft_data = res.result()
        # fft_data = np.fft.fftshift(fft_data[2], axes=1)

        # toc_fft = time.perf_counter()
        # self.logger.info(f"tidx {tidx}: FFT took {(toc_fft - tic_fft):6.4f}s. ")

        # if tidx == 1:
        #    np.savez(f"test_data/fft_array_s{tidx:04d}.npz", fft_data = fft_data)

        self.logger.info(f"task_list: Received type {type(data_chunk)} for chunk_idx {data_chunk.tb.chunk_idx}")

        for task in self.task_list:
            task.submit(self.executor_anl, data_chunk)
        # #    #task.submit(self.executor, fft_data, tidx)

# End of file tasks_mpi.py
