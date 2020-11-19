# -*- Encoding: UTF-8 -*-

"""Wavelet pre-processing."""

import logging
import numpy as np
from skimage.restoration import estimate_sigma, denoise_wavelet


class pre_wavelet():
    """Implements wavelet filtering."""

    def __init__(self, params):
        """Instantiates the pre_wavelet class as a callable.

        Args:
            params (dictionary):
                Provides keywords that are passed to `skimage.restoration.denoise_wavelet
                <https://scikit-image.org/docs/dev/api/skimage.restoration.html#denoise-wavelet>`_

        Returns:
            None
        """
        self.logger = logging.getLogger("simple")
        self.params = params

    def process(self, data_chunk, executor):
        """Executes wavelet filter on the executor.

        Args:
            data_chunk (2d image):
                Data chunk to be wavelet transformed.

            executor (PEP-3148-style executor):
                Executor on which to execute.

        Returns:
            data_chunk (2d_image):
                Wavelet-filtered images
        """
        # Execute wavelet denoising on the executor
        # fut = executor.submit()
        # Get number of channels and samples
        num_ch = data_chunk.shape[data_chunk.axis_ch]
        # num_t = data_chunk.shape[data_chunk.axis_t]

        for ch in range(num_ch):
            # Extract 1d signal 
            signal = np.take(data_chunk.data, ch, data_chunk.axis_ch)
            # Estimate sigma (on executor)
            fut = executor.submit(estimate_sigma, signal)
            sigma = fut.result()

            # Apply wavelet filtering on executor
            fut = executor.submit(denoise_wavelet, signal, sigma=sigma, **self.params)
            signal = fut.result()
            # Replace input signal with filtered signal
            np.put_along_axis(data_chunk.data, np.array([[ch]]), signal, data_chunk.axis_ch)

        return data_chunk

# End of file pre_wavelet.py
