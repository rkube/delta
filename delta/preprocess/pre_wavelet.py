# -*- Encoding: UTF-8 -*-

from skimage.restoration import estimate_sigma, denoise_wavelet


class pre_wavelet():
    """Implements wavelet filtering"""

    def __init__(self, params):
        """Instantiates the pre_wavelet class as a callable.

        Args:
            params (dictionary):
                Provides keywords that are passed to `skimage.restoration.denoise_wavelet
                <https://scikit-image.org/docs/dev/api/skimage.restoration.html#denoise-wavelet>`_

        """
        self.params = params

    def process(self, data_chunk, executor):
        # Execute wavelet denoising on the executor
        # fut = executor.submit()
        return data_chunk


# End of file pre_wavelet.py
