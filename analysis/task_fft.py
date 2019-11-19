# Coding: UTF-8 -*-


"""
This defines a task that performs a FFT of the stremed chunks for the dask application.
All helper functions are defined locally and adapted from specs.py
"""

import numpy as np
import dask.array as da
from scipy.signal import detrend, spectrogram
from math import floor



def fft_window(tnum, nfft, window, overlap):
    # IN : full length of time series, nfft, window name, overlap ratio
    # OUT : bins, 1 x nfft window function

    # use overlapping
    bins = int(np.fix((tnum // nfft - overlap) / (1.0 - overlap)))

    # window function
    if window == 'rectwin':  # overlap = 0.5
        win = np.ones(nfft)
    elif window == 'hann':  # overlap = 0.5
        win = np.hanning(nfft)
    elif window == 'hamm':  # overlap = 0.5
        win = np.hamming(nfft)
    elif window == 'kaiser':  # overlap = 0.62
        win = np.kaiser(nfft, beta=30)
    elif window == 'HFT248D':  # overlap = 0.84
        z = 2. * np.pi / nfft * np.arange(0, nfft)
        win = 1 - 1.985844164102*np.cos(z) + 1.791176438506*np.cos(2*z) - 1.282075284005*np.cos(3*z) + \
            0.667777530266*np.cos(4*z) - 0.240160796576*np.cos(5*z) + 0.056656381764*np.cos(6*z) - \
            0.008134974479*np.cos(7*z) + 0.000624544650*np.cos(8*z) - 0.000019808998*np.cos(9*z) + \
            0.000000132974*np.cos(10*z)

    return bins, win


def create_fft_freq(dt, nfft, full):
    """
    Creates a list of Fourier Frequencies
    Inputs:
    =======
    dt: float, Sampling frequency in Hz
    nfft: int, number of data points to apply FFT to
    full: bool, if True, then we want the DFT from -fN...0...fN. If false, we do a half-DFT from 0...fN

    Returns:
    ========
    freqs, ndarray(float): Frequencies of the Fourier Coefficients calculated by FFT.
    """
    freqs = da.fft.fftfreq(nfft, d=dt) 

    if (nfft % 2 == 0):
        freqs = da.hstack([freqs[0: nfft // 2], -freqs[nfft //2], freqs[nfft // 2:nfft]])
    if full:
        freqs = da.fft.fftshift(freqs)
    else:
        freqs = freqs[0:nfft // 2 + 1]

    return(freqs)


def allocate_fft_data(nfft, bins, full):
    """
    Allocate data to store DFT in.
    Inputs:
    =======
    nfft: int, number of data points to apply FFT to
    full: bool, if True, then we want the DFT from -fN...0...fN. If false, we do a half-DFT from 0...fN
    bins: Number of FFTs per channel
    """

    if full: # full shift to -fN ~ 0 ~ fN
        if (nfft % 2) == 0:  # even nfft
            # TODO: Store everything as float32/complex64
            fftdata = da.zeros((bins, nfft + 1), dtype=np.complex128)
        else:  # odd nfft
            # TODO: Store everything as float32/complex64
            fftdata = da.zeros((bins, nfft), dtype=np.complex128)
    else: # half 0 ~ fN
        # TODO: Store everything as float32/complex64
        fftdata = da.zeros((bins, nfft // 2 + 1), dtype=np.complex128)

    return fftdata



class task_fft():
    """Performs a DFT for incoming data chunks."""
    def __init__(self, data_per_chunk, fft_params, normalize=True, detrend=True):
        """
        Inputs:
        =======
        data_per_chunk: Number of data points per chunk (10_000)
        fft_params: dictionary for fft parameters
        normalize: bool, If true, normalize data before applying FFTs.
        detrend: bool, If true, detrend data before applying FFTs.
        """

        self.ndata = data_per_chunk
        # nfft: Number of data points per fft
        self.nfft = fft_params["nfft"]
        # Type of filter window
        self.window = fft_params["window"]
        # Amount of overlap for successive DFTs
        self.overlap = fft_params["overlap"]
        # Detrend data or not
        self.detrend = fft_params["detrend"]
        # Sampling frequency of the data
        self.fs = fft_params["fsample"]
        self.full = True

        # Calculate number of bins (ffts per channel) and the windowing function
        self.fft_bins, self.fft_win = fft_window(self.ndata, self.nfft, self.window, self.overlap)
        # Calculate the frequencies
        self.fft_freqs = create_fft_freq(self.fs, self.nfft, self.full)

    def do_fft(self, cs_data):
        """Perform the FFT. This former fftbins routine
        Inputs:
        cs_data: dask array(float). dim0: channels(192) dim1: samples (10_000)
        """

        for fbin in range(self.fft_bins):
            idx1 = int(fbin * floor(nfft * (1. - overlap)))
            idx2 = idx1 + nfft

            sx = cs_data[:, idx1:idx2]

            if detrend == 1:
                sx = detrend(sx, type='linear', axis=1)
            sx = detrend(sx, type='constant', axis=1)  # subtract mean

            sx = sx * win  # apply window function

            # get fft
            SX = np.fft.fft(sx, n=nfft)/nfft  # divide by the length
            if np.mod(nfft, 2) == 0:  # even nfft
                SX = np.hstack([SX[0:int(nfft/2)], np.conj(SX[int(nfft/2)]), SX[int(nfft/2):nfft]])
            if full == 1: # shift to -fN ~ 0 ~ fN
                SX = np.fft.fftshift(SX)
            else: # half 0 ~ fN
                SX = SX[0:int(nfft/2+1)]

            fftdata[b,:] = SX

        return ax, fftdata, win_factor


class task_fft_scipy():
    """Performs a DFT for incoming data chunks using methods from scipy.signal."""
    def __init__(self, data_per_chunk, fft_params, normalize=True, detrend=True):
        """
        Inputs:
        =======
        data_per_chunk: Number of data points per chunk (10_000)
        fft_params: dictionary for fft parameters
        normalize: bool, If true, normalize data before applying FFTs.
        detrend: bool, If true, detrend data before applying FFTs.
        """

        print("init: fft_params = ", fft_params)

        self.ndata = data_per_chunk
        # nfft: Number of data points per fft
        self.nfft = fft_params["nfft"]
        # Type of filter window
        self.window = fft_params["window"]
        # Amount of overlap for successive DFTs
        self.overlap = fft_params["overlap"]
        # Detrend data or not
        self.detrend = fft_params["detrend"]
        # Sampling frequency of the data
        self.fs = fft_params["fsample"]
        self.full = True

        self.noverlap = int(self.nfft * self.overlap)


    def do_fft(self, dask_client, raw_data):
        
        def stft_scipy(data_in, idx):
            res = spectrogram(data_in, nfft=self.nfft, window=self.window,
                              nperseg=self.nfft,
                              detrend="linear", noverlap=self.noverlap,
                              scaling="spectrum", mode="complex", return_onesided=False)
            return res[2]

        futures = dask_client.map(stft_scipy, raw_data, range(2))
        results = dask_client.gather(futures)

        return(results)


# End of file task_fft.py