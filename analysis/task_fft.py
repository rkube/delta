# Coding: UTF-8 -*-


"""
Author: Ralph Kube
This defines a task that performs a FFT of the stremed chunks for the dask application.
All helper functions are defined locally and adapted from specs.py anf fluctana.py

This class is not used for the mpi implementation and can be deleted.
"""

import numpy as np
from scipy.signal import detrend, spectrogram, stft
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



class task_fft_kstar():
    """Performs a DFT for incoming data chunks. This class uses the methods provided in
    the fluctana package for the DFTs."""
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

        self.fft_window = fft_window(self.ndata, self.nfft, self.window, self.overlap)

    def do_fft(self, cs_data):
        """Perform the FFT. This former fftbins routine
        Inputs:
        cs_data: dask array(float). dim0: channels(192) dim1: samples (10_000)
        """

        for fbin in range(self.fft_bins):
            idx1 = int(fbin * floor(self.nfft * (1. - self.overlap)))
            idx2 = idx1 + self.nfft

            sx = cs_data[:, idx1:idx2]

            if detrend == 1:
                sx = detrend(sx, type='linear', axis=1)
            sx = detrend(sx, type='constant', axis=1)  # subtract mean

            sx = sx * self.fft_window  # apply window function

            # get fft
            SX = np.fft.fft(sx, n=self.nfft)/self.nfft  # divide by the length
            if np.mod(self.nfft, 2) == 0:  # even nfft
                SX = np.hstack([SX[0:self.nfft // 2], np.conj(SX[self.nfft //2 ]), SX[self.nfft // 2:self.nfft]])
            if full == 1: # shift to -fN ~ 0 ~ fN
                SX = np.fft.fftshift(SX)
            else: # half 0 ~ fN
                SX = SX[0:self.nfft//2 + 1]

            fftdata[fbin,:] = SX

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
        self.full = fft_params["full"]

        self.noverlap = int(self.nfft * self.overlap)

        _, win = self.build_fft_window(self.ndata, self.nfft, self.window, self.overlap)
        self.win_factor = np.mean(win**2.0)

        
    def build_fft_window(self, tnum, nfft, window, overlap):
        """Builds the window used in the STFTs. Taken from KSTAR/specs.py

        Input:
        ======
        tnum: int, length of the input time series
        nfft: int, data points used in FFT
        window: string, defines the window to use. See corresponding numpy functions.
        overlap: float, Overlap between ffts, relative to nfft.

        Returns:
        ========
        bins: int, The number of individual data points performed on the input time series
        win: ndarray, flot, The window function applied to input-segments of the FFT
        """

        # use overlapping
        bins = int(np.fix((int(tnum/nfft) - overlap)/(1.0 - overlap)))

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
            z = 2*np.pi/nfft*np.arange(0,nfft)
            win = 1 - 1.985844164102*np.cos(z) + 1.791176438506*np.cos(2*z) - 1.282075284005*np.cos(3*z) + \
                0.667777530266*np.cos(4*z) - 0.240160796576*np.cos(5*z) + 0.056656381764*np.cos(6*z) - \
                0.008134974479*np.cos(7*z) + 0.000624544650*np.cos(8*z) - 0.000019808998*np.cos(9*z) + \
                0.000000132974*np.cos(10*z)

        return bins, win


    def get_fft_params(self):
        """Returns parameters of the fft in dictionary form."""

        fft_params = {"nfft": self.nfft, "window": self.window, "overlap": self.overlap,
                      "detrend": self.detrend, "fs": self.fs, "noverlap": self.noverlap,
                      "win_factor": self.win_factor, "full": self.full}

        return fft_params



    def do_fft(self, executor, stream_data):
        """Dispatch a STFT to the workers.
        For details on how the STFT relates to other spectrograms, see
        tests_div/scipy_compare_spectrograms.ipynb

        The spectrogram is computed using
        *self.nfft points per fft
        *self.window as the windowing function
        *self.nfft as nperseg
        *self.noverlap = int(self.nfft * self.overlap) the overlap
        linear detrending

        The returned spectrogram is the average of all calculated fourier transformations.

        Input:
        ======
        executor: PEP-3148 style executor
        stream_data_future:

        Returns:
        ========

        List of futures
        """

        def stft_scipy(data_in):
            """ Calculates short-time fourier transformations using scipy.signal.spectrogram
            Inputs
            ======
            data_in: ndarray, float. Time-data to be Fourier-Transformed. dim0: channel, dim1: time
            ch_idx: int, Index of the channel to apply the STFT to.

            Returns:
            ========
            res[2]: The third element of the return-tuple from spectrogram.
                    ndarray, complex dim0: Fourier Coefficients. dim1: index of the n-th stft.
            """

            res = stft(data_in, axis=1, fs=self.fs, nperseg=self.nfft, window=self.window,
                       detrend=self.detrend, noverlap=self.noverlap, padded=False,
                       return_onesided=False, boundary=None)

            res = np.fft.fftshift(res, axes=1)
            
            return res[2]

        # Distribute the stft function to the workers
        futures = [executor.submit(stft_scipy, stream_data)]
        return(futures)

    def do_fft_local(self, stream_data):
        """Performs STFT locally"""

        res = stft(stream_data, axis=1, fs=self.fs, nperseg=self.nfft, window=self.window,
                   detrend=self.detrend, noverlap=self.noverlap, padded=False,
                   return_onesided=False, boundary=None)

        res = np.fft.fftshift(res[2], axes=1)

        return res


# End of file task_fft.py