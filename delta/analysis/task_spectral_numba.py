# -*- Encoding: UTF-8 -*-

from analysis.task_base import task_base
from analysis.kernels_spectral_gpu import kernel_spectral_GAP, increment_by_one, increment_by_two


def calc_and_store_numba(kernel, storage_backend, fft_data, ch_it, info_dict):
    """Dispatches a GPU numba kernel and store the result.

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
    import numpy as np
    import math

    comm = MPI.COMM_WORLD

    # Code below tests dummy kernel
    # out_arr = np.zeros(100)
    # threadsperblock = 32
    # blockspergrid = (out_arr.size + (threadsperblock - 1)) // threadsperblock
    # kernel[blockspergrid, threadsperblock](out_arr)
    # End test of dummy kernel

    result = np.zeros([len(ch_it), fft_data.data.shape[1], 3], dtype=fft_data.data.dtype)
    
    threads_per_block = (32, 32)
    num_blocks = [math.ceil(s / t) for s, t in zip(result.shape, threads_per_block)]
    ch1_idx_arr = np.array([c.ch1.get_idx() for c in ch_it])
    ch2_idx_arr = np.array([c.ch2.get_idx() for c in ch_it])
    win_factor = 1.0

    # Try changing flags to C_CONTIGUOUS
    # Passing fft_data.data directly into the kernel always fails.
    # I checked size and passing a dummy array of similar shape and dtype.
    # That worked, buy never fft_data.data
    # I also checked the flags. fft_data.data.C_CONTIGUOUS was false. Setting it to true
    # also didn't allow me to pass this into the kernel.
    # Now I'm doing this here:
    dummy = np.zeros(fft_data.data.shape, dtype=fft_data.data.dtype)
    dummy[:] = fft_data.data[:]

    t1_calc = datetime.datetime.now()
    kernel[num_blocks, threads_per_block](dummy, result, ch1_idx_arr, ch2_idx_arr, win_factor)

    t2_calc = datetime.datetime.now()

    t1_io = datetime.datetime.now()
    storage_backend.store_data(result, info_dict)
    dt_io = datetime.datetime.now() - t1_io

    with open(f"outfile_{comm.rank:03d}.txt", "a") as df:
        # df.write(f"success: num_blocks={num_blocks}, tpb={threads_per_block}... {fft_data.data.dtype}, {fft_data.data.shape}... ")
        # df.write(f"dummy: {dummy.flags}, fft_data.data: {fft_data.data.flags}")
        df.write((f"rank {comm.rank:03d}/{comm.size:03d}: "
                  f"tidx={info_dict['chunk_idx']} {info_dict['analysis_name']} "
                  f"start {t1_calc.isoformat(sep=' ')}  "
                  f"end {t2_calc.isoformat(sep=' ')} "
                  f"Storage: {dt_io} {gethostname()}\n"))
        df.flush()

    return None


class task_spectral_GAP(task_base):
    """Calculates coherence, cross-power and cross-phase in a fused GPU kernel."""
    def _get_kernel(self):
        return kernel_spectral_GAP
        #return increment_by_two
        #return None

    def __str__(self):
        return "task_spectral_GAP"

    def _get_dispatch_func(self):
        return calc_and_store_numba

# End of file task_spectral_numba.py
