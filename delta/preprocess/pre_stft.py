# -*- Encoding: UTF-8 -*-

import numpy as np
from scipy.signal import stft


class pre_stft():
    """Implements short-time Fourier transformation"""

    def __init__(self, params):
        """Instantiates the STFT class as a callable.

        Args:
            params (dictionary):
                Provides keywords that are passed to `scipy.signal.stft
                <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html>`_

        """
        self.params = params
        self.params["noverlap"] = int(self.params["overlap"] * self.params["nfft"])

    def process(self, data_chunk, executor):
        """Performs STFT preprocessing on an executor

        Args:
            data_chunk (twod_chunk):
                Time-chunk of image data.
            executor (PEP-3148 executor):
                Executor on which calls to submit will be launched

        Returns:
            data_chunk_ft (twod_chunk_f):
                Fourier-transformed image-data chunk
        """

        fut = executor.submit(stft, data_chunk.data(), axis=data_chunk.axis_t,
                              fs=self.params["fs"], nperseg=self.params["nfft"],
                              window=self.params["window"],
                              detrend=self.params["detrend"],
                              noverlap=self.params["noverlap"],
                              padded=False,
                              return_onesided=False,
                              boundary=None)
        data_fft = fut.result()
        data_fft = np.fft.fftshift(data_fft[2], axes=data_chunk.axis_t)

        # Calculate the windowing factor and add it to the parameters
        _, win = self.build_fft_window(data_chunk.data().shape[data_chunk.axis_t],
                                       self.params["nfft"],
                                       self.params["window"], self.params["overlap"])
        self.params["win_factor"] = (win ** 2.0).mean()

        data_chunk_ft = data_chunk.create_ft(data_fft, self.params)
        return data_chunk_ft

    def build_fft_window(self, tnum, nfft, window, overlap):
        """Builds the window used in the STFTs. Taken from KSTAR/specs.py

        Args:
            tnum (int):
                Number of samples in the input array
            nfft (int):
                Number of data points used in FFT
            window (str):
                Defines the window to use. See corresponding numpy functions.
            overlap (float)
                Overlap between ffts as a fraction of tnum. Between 0 and 1.

        Returns:
            bins (int):
                The number of individual data points performed on the input time series
            win (ndarray):
                The window function applied to input-segments of the FFT
        """

        assert((overlap > 0.0) & (overlap < 1.0))

        # use overlapping
        bins = int(np.fix((int(tnum / nfft) - overlap) / (1.0 - overlap)))

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
            win = 1. - 1.985844164102 * np.cos(z) +\
                1.791176438506 * np.cos(2 * z) -\
                1.282075284005 * np.cos(3 * z) +\
                0.667777530266 * np.cos(4 * z) -\
                0.240160796576 * np.cos(5 * z) +\
                0.056656381764 * np.cos(6 * z) -\
                0.008134974479 * np.cos(7 * z) +\
                0.000624544650 * np.cos(8 * z) -\
                0.000019808998 * np.cos(9 * z) +\
                0.000000132974 * np.cos(10 * z)
        else:
            raise NameError(f"Unknown FFT window name: {window}")

        return bins, win


# End of file pre_stft.py
