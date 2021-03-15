# -*- Encoding: UTF-8 -*-
"""Defines task objects that calculate spectra coherence."""

import logging

from data_models.helpers import get_dispatch_sequence
from storage.backend import get_storage_object


def calc_and_store(kernel, storage_backend, timechunk, ch_it, info_dict):
    """Dispatch a kernel and store the result.

    Args:
        timechunk (ecei_chunk_ft):
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
    result = kernel(timechunk.data_ft, ch_it, timechunk.params)
    with open(f"outfile_{(comm.rank):03d}.txt", "a") as df:
        df.write(f"{an_name}: type is {type(result)} and shape is {result.shape}\n")
        df.flush()

    t2_calc = datetime.datetime.now()

    t1_io = datetime.datetime.now()
    storage_backend.store_data(result, info_dict)
    dt_io = datetime.datetime.now() - t1_io

    with open(f"outfile_{(comm.rank):03d}.txt", "a") as df:
        # df.write("here")
        df.write(f"rank {comm.rank:03d}/{comm.size:03d}: tidx={chunk_idx} {an_name} start " +
                t1_calc.isoformat(sep=" ") + " end " + t2_calc.isoformat(sep=" ") +
                f" Storage: {dt_io} " + f" {gethostname()}\n")
        df.write(f"{an_name}: type is {type(result)} and shape is {result.shape}\n")
        df.flush()

    return result


class task_base():
    """Task object to calculate the coherence."""
    def __init__(self, params, cfg_storage):
        """Initializes task_coherence."""
        self.logger = logging.getLogger("simple")
        self.params = params
        # TODO: dispatch sequence is still hard-coded to suit ECEI analysis.
        #       Re-factor this to only apply to ECEI data.
        #       Especially, we need to serialize the dispatch sequence so that it
        #       can be stored together with the parameters in the call to store_metadata.
        self.dispatch_seq = get_dispatch_sequence(self.params["ref_channels"],
                                                  self.params["cmp_channels"],
                                                  self.params["channel_chunk_size"])
        storage_class = get_storage_object(cfg_storage)
        self.storage_backend = storage_class(cfg_storage)
        self.storage_backend.store_metadata(params)

    def _get_kernel(self):
        """Returns the analysis kernel to launch."""
        return None

    def _get_dispatch_func(self):
        """Returns the dispatch function to use."""
        return calc_and_store

    def execute(self, timechunk, executor):
        """Launches a spectral analysis kernel on an executor.

        Args:
            executor (`PEP-3148 <https://www.python.org/dev/peps/pep-3148/>`_ compatible executor):
                Executor to use
            fft_data (data-model):
                Fourier Coefficients of the data to analyze

        Returns:
            None
        """
        info_dict_list = [{"analysis_name": self.__str__(),
                           "chunk_idx": timechunk.tb.chunk_idx,
                           "channel_batch": batch_idx}
                          for batch_idx in range(len(self.dispatch_seq))]

        _ = [executor.submit(self._get_dispatch_func(),
                             self._get_kernel(),
                             self.storage_backend,
                             timechunk,
                             ch_it,
                             info_dict) for ch_it, info_dict in zip(self.dispatch_seq,
                                                                    info_dict_list)]
        self.logger.info((f"chunk_idx={timechunk.tb.chunk_idx} submitted {self.__str__()} "
                          f"as {len(self.dispatch_seq)} tasks: {self._get_kernel()} "
                          f"dispatch_function: {self._get_dispatch_func()}"))
        return None

# End of file task_base.py
