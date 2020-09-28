# -* Encoding: UTF-8 -*-

import numpy as np
import cupy as cp

def kernel_null(fft_data, ch_it, fft_config):
    """Does nothing. Used in performance testing to evaluate framework communication overhead"""
    return(None)


def kernel_crossphase_cu(fft_data, ch_it, fft_config):
    """Kernel that calculates the cross-phase between two channels.
    Parameters:
    -----------
    fft_data: ndarray, complex:
              Holds the fourier-transformed data.
              dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it...: iterable,
              Iterator over a list of channels we wish to perform our computation on

    Returns:
    ========
    Axy: float, the cross phase
    """


    fft_data_cu = cp.asarray(fft_data)
    # Gather results on device, since
    Pxy = cp.zeros([len(ch_it), fft_data.shape[1]], dtype=fft_data.dtype)

    # The result of the call to mean() is another cupy array. Gather the
    # results in Pxy which needs to by a cupy array. Copy to host once
    # loop is done
    for idx, ch_pair in enumerate(ch_it):
        Pxy[idx, :] = (fft_data_cu[ch_pair.ch1.get_idx(), :, :] * fft_data_cu[ch_pair.ch2.get_idx(), :, :].conj()).mean(axis=1)

    Pxy = cp.asnumpy(Pxy)

    return(np.arctan2(Pxy.imag, Pxy.real).real)



def kernel_crosspower_cu(fft_data, ch_it, fft_config):
    """Kernel that calculates the cross-power between two channels.
    Parameters:
    -----------
    fft_data: ndarray, float: Contains the fourier-transformed data.
                dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it: iterable, Iterator over a list of channels we wish to perform our computation on


    Returns:
    --------
    cross_power, float.
    """
    res = cp.zeros([len(ch_it), fft_data.shape[1]], dtype=fft_data.dtype)
    for idx, ch_pair in enumerate(ch_it):
        res[idx, :] = (fft_data[ch_pair.ch1.get_idx(), :, :] * fft_data[ch_pair.ch2.get_idx(), :, :].conj()).mean(axis=1) / fft_config["fft_params"]["win_factor"]

    res = cp.asnumpy(res)

    return(np.abs(res).real)


def kernel_crosscorr_cu(fft_data, ch_it, fft_params):
    """Defines a kernel that calculates the cross-correlation between two channels.

    Parameters:
    ----------
    fft_data: ndarray, float: Contains the fourier-transformed data.
                dim0: channel. dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it: iterable, Iterator over a list of channels we wish to perform our computation on
    fft_params: dict, parameters of the fourier-transformed data


    Returns:
    --------
    cross-correlation, float array
    """

    import time
    from cupyx.scipy import fft

    fft_shifted = np.fft.fftshift(fft_data, axes=1)

    fft_shifted_gpu = cp.asarray(fft_shifted)
    res = cp.zeros([len(ch_it), fft_data.shape[1]])

    tic_list = []
    toc_list = []

    # Pre-calculate one plan for all following FFTs:
    plan = fft.get_fft_plan(fft_shifted_gpu[0, :, :], axes=0)
    with plan:
        for idx, ch_pair in enumerate(ch_it):
            tic_list.append(time.perf_counter())
            # _tmp = cp.fft.ifft(fft_shifted_gpu[1, :, :] *
            #                    fft_shifted_gpu[3, :, :].conj(),
            #                    axis=0).mean(axis=1) / fft_params['win_factor']

            _tmp = cp.fft.ifft(fft_shifted_gpu[ch_pair.ch1.get_idx(), :, :] *
                              fft_shifted_gpu[ch_pair.ch2.get_idx(), :, :].conj(),
                              axis=0).mean(axis=1) / fft_params['win_factor']
            toc_list.append(time.perf_counter())
            res[idx, :] = cp.fft.fftshift(_tmp.real)

    dts = [f"{toc - tic}" for toc, tic in zip(toc_list, tic_list)]
    with open("cupy_timings.txt", "a") as df:
        df.write("\n".join(dts) + "\n")

    res = cp.asnumpy(res)

    return(res)



# End of file kernels_spectral_cu.py
