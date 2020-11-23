# -*- Encoding: UTF-8 -*-

"""This file contains analysis kernels for spectral (fourier-transformed) data.

All kernels have a more-or-less uniform interface
"""

import numpy as np


def kernel_null(data, ch_it, fft_config):
    """Does nothing.

    Used for performance testing.
    """
    import numpy as np

    with open("null.txt", "a") as df:
        df.write("kernel_null executed\n")
        df.flush()

    np.savez("kernel_null.npz", data=data)
    return(data)


def kernel_crossphase(fft_data, ch_it, fft_config):
    """Kernel that calculates the cross-phase between two channels.

    Args:
        fft_data (ndarray, complex):
          Contains the fourier-transformed data.
          dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
        ch_it (iterable):
          Iterator over a list of channels we wish to perform our computation on

    Returns:
      Axy (float):
        Cross phase
    """
    Pxy = np.zeros([len(ch_it), fft_data.shape[1]], dtype=fft_data.dtype)
    for idx, ch_pair in enumerate(ch_it):
        Pxy[idx, :] = (fft_data[ch_pair.ch1.get_idx(), :, :] *
                       fft_data[ch_pair.ch2.get_idx(), :, :].conj()).mean(axis=1)

    crossphase = np.zeros_like(Pxy.real)
    np.arctan2(Pxy.imag, Pxy.real, out=crossphase, where=(Pxy.real > 1e-6) | (Pxy.imag > 1e-6))

    return crossphase


def kernel_crosspower(fft_data, ch_it, fft_config):
    """Kernel that calculates the cross-power between two channels.

    Args:
    fft_data (ndarray, complex):
        Contains the fourier-transformed data.
        dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it (iterable):
        Iterator over a list of channels we wish to perform our computation on


    Returns:
        cross_power (float):
            Cross-power
    """
    res = np.zeros([len(ch_it), fft_data.shape[1]], dtype=fft_data.dtype)
    for idx, ch_pair in enumerate(ch_it):
        res[idx, :] = (fft_data[ch_pair.ch1.get_idx(), :, :] *
                       fft_data[ch_pair.ch2.get_idx(), :, :].conj()).mean(axis=1) /\
            fft_config["win_factor"]

    return(np.abs(res).real)


def kernel_coherence(fft_data, ch_it, fft_config):
    """Kernel that calculates the coherence between two channels.

    Args:
    fft_data (ndarray, complex):
        Contains the fourier-transformed data.
        dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it (iterable):
        Iterator over a list of channels we wish to perform our computation on

    Returns:
        coherence (float)
    """
    Gxy = np.zeros([len(ch_it), fft_data.shape[1]], dtype=fft_data.dtype)

    for idx, ch_pair in enumerate(ch_it):
        X = fft_data[ch_pair.ch1.get_idx(), :, :]
        Y = fft_data[ch_pair.ch2.get_idx(), :, :]
        Pxx = X * X.conj()
        Pyy = Y * Y.conj()
        Gxy[idx, :] = np.abs(X * Y.conj() / np.sqrt(Pxx * Pyy)).mean(axis=1)

    Gxy = Gxy.real
    return(Gxy)


def kernel_crosscorr(fft_data, ch_it, fft_params):
    """Defines a kernel that calculates the cross-correlation between two channels.

    Args:
    fft_data (ndarray, complex):
        Contains the fourier-transformed data.
        dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it (iterable):
        Iterator over a list of channels we wish to perform our computation on

    Returns:
        cross-correlation (float)
    """
    res = np.zeros([len(ch_it), fft_data.shape[1]])
    fft_shifted = np.fft.fftshift(fft_data, axes=1)

    for idx, ch_pair in enumerate(ch_it):
        X = fft_shifted[ch_pair.ch1.get_idx(), :, :]
        Y = fft_shifted[ch_pair.ch2.get_idx(), :, :]
        _tmp = np.fft.ifft(X * Y.conj(), axis=0).mean(axis=1) / fft_params['win_factor']

        res[idx, :] = np.fft.fftshift(_tmp.real)

    with open("/global/homes/r/rkube/repos/delta/delta/xcorr.txt", "a") as df:
        df.write("kernel_crosscorr executed\n")
        df.flush()

    return(res)


def kernel_bicoherence(fft_data, ch_it, fft_params):
    """Kernel that calculates the bi-coherence between two channels.

    Args:
    fft_data (ndarray, complex):
        Contains the fourier-transformed data.
        dim0: channel, dim1: Fourier Coefficients, dim2: STFT (bins in fluctana code)
    ch_it (iterable):
        Iterator over a list of channels we wish to perform our computation on

    Returns:
        bicoherence (float)
    """
    res_list = []

    for ch_pair in ch_it:
        ch1_idx, ch2_idx = ch_pair.ch1.get_idx(), ch_pair.ch2.get_idx()

        # Transpose to make array layout compatible with code from specs.py
        XX = np.fft.fftshift(fft_data[ch1_idx, :, :], axes=0).T
        YY = np.fft.fftshift(fft_data[ch2_idx, :, :], axes=0).T

        bins, full = XX.shape
        half = full // 2 + 1

        # calculate bicoherence
        B = np.zeros((full, half), dtype=np.complex_)
        P12 = np.zeros((full, half))
        P3 = np.zeros((full, half))
        val = np.zeros((full, half))

        for b in range(bins):
            X = XX[b, :]  # full -fN ~ fN
            Y = YY[b, :]  # full -fN ~ fN

            Xhalf = np.fft.ifftshift(X)   # full 0 ~ fN, -fN ~ -f1
            Xhalf = Xhalf[0:half]         # half 0 ~ fN

            X1 = np.transpose(np.tile(X, (half, 1)))
            X2 = np.tile(Xhalf, (full, 1))
            X3 = np.zeros((full, half), dtype=np.complex_)
            for j in range(half):
                if j == 0:
                    X3[0:, j] = Y[j:]
                else:
                    X3[0:(-j), j] = Y[j:]

            B = B + X1 * X2 * np.matrix.conjugate(X3) / bins  # complex bin average
            P12 = P12 + (np.abs(X1 * X2).real)**2 / bins      # real average
            P3 = P3 + (np.abs(X3).real)**2 / bins             # real average

        val = (np.abs(B)**2) / P12 / P3   # bicoherence

        # summation over pairs
        sum_val = np.zeros(full)
        for i in range(half):
            if i == 0:
                sum_val = sum_val + val[:, i]
            else:
                sum_val[i:] = sum_val[i:] + val[:-i, i]

        N = np.array([i + 1 for i in range(half)] + [half for i in range(full - half)])
        sum_val = sum_val / N   # element wise division

        res_list.append((val, sum_val))

    return (res_list)


def kernel_skw(fft_data, ch_it, fft_params, ecei_config, kstep=0.01):
    """Calculates the conditional spectrum S(k,w).

    Input:
        fft_data (ndarray, cmplx):
            Contains the fourier-transformed data. dim0: channel,
            dim1: Fourier Coefficients
        ch_it (iterable):
            Iterator over a list of channels we wish to perform our computation on
        fft_params (dictionary):
            Parameters of the fft
        ecei_config (dictionary):
            configuration of ecei diagnostic

    Returns:
        bicoherence (float)
    """
    from analysis.ecei_helper import channel_position

    res_list = []
    for ch_pair in ch_it:

        ch1 = ch_pair.ch1
        ch2 = ch_pair.ch2
        ch1_idx, ch2_idx = ch1.get_idx(), ch2.get_idx()

        nfft = int(fft_params["nfft"])

        if(ch1_idx == ch2_idx):
            # We can't calculate the cross-conditional spectrum for ch0==ch1
            res_list.append(None)
            continue
        XX = np.fft.fftshift(fft_data[ch1_idx, :, :], axes=0).T
        YY = np.fft.fftshift(fft_data[ch2_idx, :, :], axes=0).T

        bins, _ = XX.shape
        win_factor = fft_params["win_factor"]

        cpos_ref = channel_position(ch1, ecei_config)
        cpos_cmp = channel_position(ch2, ecei_config)

        # Calculate distance between channels
        dist = np.sqrt((cpos_ref[0] - cpos_cmp[0])**2.0 + (cpos_ref[1] - cpos_cmp[1])**2.0)
        dmin = dist * 1e2

        kax = np.arange(-np.pi / dmin, np.pi / dmin, kstep)

        nkax = kax.size

        # value dimension
        Pxx = np.zeros((bins, nfft), dtype=np.complex_)
        Pyy = np.zeros((bins, nfft), dtype=np.complex_)
        Kxy = np.zeros((bins, nfft), dtype=np.complex_)
        val = np.zeros((nkax, nfft), dtype=np.complex_)

        sklw = np.zeros((nkax, nfft), dtype=np.complex_)
        K = np.zeros(nfft, dtype=np.complex_)
        sigK = np.zeros(nfft, dtype=np.complex_)

        # logging.info(f"nkax = {nkax:d}, nfft = {nfft:d}, dmin = {dmin:f}")

        # calculate auto power and cross phase (wavenumber)
        for b in range(bins):
            X = XX[b, :]
            Y = YY[b, :]

            Pxx[b, :] = X * np.matrix.conjugate(X) / win_factor
            Pyy[b, :] = Y * np.matrix.conjugate(Y) / win_factor
            Pxy = X * np.matrix.conjugate(Y)
            Kxy[b, :] = np.arctan2(Pxy.imag, Pxy.real).real / (dist * 100)  # [cm^-1]

            # calculate SKw
            for w in range(nfft):
                idx = (Kxy[b, w] - kstep * 0.5 < kax) * (kax < Kxy[b, w] + kstep * 0.5)
                val[:, w] = val[:, w] + (1.0 / bins * (Pxx[b, w] + Pyy[b, w]) * 0.5) * idx

        # calculate moments
        sklw = val / np.tile(val.sum(axis=0), (nkax, 1))
        K[:] = np.sum(np.transpose(np.tile(kax, (nfft, 1))) * sklw, axis=0)
        for w in range(nfft):
            sigK[w] = np.sqrt(np.sum((kax - K[w])**2 * sklw[:, w]))

        val = val.mean(axis=0).real
        K = np.mean(K, axis=0)
        sigK = np.mean(sigK, axis=0)
        pdata = np.log10(val + 1e-10)

        res_list.append(pdata)

    return(res_list)


# End of file kernels_spectral.py
