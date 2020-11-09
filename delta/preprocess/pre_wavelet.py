# -*- Encoding: UTF-8 -*-

"""Wavelet pre-processing."""

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

        #sigma = estimate_sigma(data_chunk)
        #denoised = denoise_wavelet(sigma)
        #denoised += 0

        return data_chunk


# End of file pre_wavelet.py
