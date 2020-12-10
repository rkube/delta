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

# Import plain python kernels
# from analysis.kernels_spectral import kernel_null, kernel_crosspower
# from analysis.kernels_spectral import kernel_crossphase, kernel_coherence kernel_crosscorr
# from analysis.kernels_spectral import kernel_bicoherence, kernel_skw
# Import cython kernels
from analysis.kernels_spectral import kernel_null, kernel_crosscorr
from analysis.kernels_spectral_cy import kernel_coherence_64_cy
from analysis.kernels_spectral_cy import kernel_crosspower_64_cy, kernel_crossphase_64_cy
from analysis.kernels_spectral_gpu import kernel_spectral_GAP
# Import CUDA kernels
# from analysis.kernels_spectral_cu import kernel_crossphase_cu, kernel_crosscorr_cu

from data_models.helpers import get_dispatch_sequence
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
    chunk_idx = info_dict['chunk_idx']
    an_name = info_dict["analysis_name"]
    t1_calc = datetime.datetime.now()
    result = kernel(fft_data.data, ch_it, fft_data.params)
    t2_calc = datetime.datetime.now()
    dt_calc = t2_calc - t1_calc

    t1_io = datetime.datetime.now()

    storage_backend.store_data(result, info_dict)
    t2_io = datetime.datetime.now()
    dt_io = t2_io - t1_io

    with open(f"outfile_{(comm.rank):03d}.txt", "a") as df:
        df.write(f"rank {comm.rank:03d}/{comm.size:03d}: tidx={chunk_idx} {an_name} start " +
                 t1_calc.isoformat(sep=" ") + " end " + t2_calc.isoformat(sep=" ") +
                 f" Storage: {dt_io} " + f" {gethostname()}\n")
        df.flush()

    return result


class task_spectral():
    """Serves as the super-class for analysis methods."""

    def __init__(self, task_config, cfg_storage):
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
        self.cfg_storage = cfg_storage
        self.logger = logging.getLogger("simple")
        self.logger.info(f"task_spectral: task_config={task_config}")

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
        elif self.analysis == "null":
            self.kernel = kernel_null
        elif self.analysis == "spectral_GAP_gpu":
            self.kernel = kernel_spectral_GAP
        else:
            raise NameError(f"Unknown analysis task {self.analysis}")

        # Number of channel pairs per future
        self.channel_chunk_size = task_config["channel_chunk_size"]
        # Total number of chunks, i.e. number of futures appended to the list per call to calculate
        #self.num_chunks = (len(self.unique_channels) +
        #                   self.channel_chunk_size - 1) // self.channel_chunk_size


        #
        self.dispatch_seq = get_dispatch_sequence(self.task_config["ref_channels"],
                                                  self.task_config["cmp_channels"],
                                                  self.task_config["channel_chunk_size"])
        self.logger.info(f"dispatch_seq = {len(self.dispatch_seq[0])}")

        # Initialize storage
        storage_class = get_storage_object(self.cfg_storage)
        self.storage_backend = storage_class(self.cfg_storage)
        self.storage_backend.store_metadata(self.task_config, self.dispatch_seq)

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
                           "channel_batch": batch_idx} for batch_idx in range(len(self.dispatch_seq))]

        _ = [executor.submit(calc_and_store,
                             self.kernel,
                             self.storage_backend,
                             fft_data,
                             ch_it,
                             info_dict) for ch_it, info_dict in zip(self.dispatch_seq,
                                                                    info_dict_list)]
        self.logger.info(f"chunk_idx={fft_data.tb.chunk_idx} submitted {self.analysis} " +
                         f"as {len(self.dispatch_seq)} tasks: {self.kernel}.  fft_data: {type(fft_data)}")
        return None


class task_list():
    """Defines interface to execute a group of tasks on an PEP-3148 executor."""

    def __init__(self, executor_anl, cfg_tasklist, cfg_storage):
        """Initialize the object with a list of tasks to be performed.

        These tasks share a common channel list.

        Args:
            executor_anl (PEP-3148 executor):
                Executor on which analysis tasks are performed
            cfg_storage (dict,):
                Storage section of the Delta config

        Returns:
            None
        """
        self.executor_anl = executor_anl
        self.logger = logging.getLogger("simple")

        self.task_list = []
        for cfg_task in cfg_tasklist:
            self.task_list.append(task_spectral(cfg_task, cfg_storage))

    def submit(self, data_chunk):
        """Launches the analysis pipeline with a data chunk.

        Args:
            data (data_chunk, data_model):
                A time chunk of data. This is a data_model derived class.

        Returns:
            None
        """

        self.logger.info(f"task_list: Received type {type(data_chunk)}\
            for chunk_idx {data_chunk.tb.chunk_idx}")


        for task in self.task_list:
            task.submit(self.executor_anl, data_chunk)

# End of file tasks_mpi.py
