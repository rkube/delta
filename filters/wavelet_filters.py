# -*- Encoding: UTF-8 -*-

from skimage.restoration import estimate_sigma, denoise_wavelet



class filter():
    """Abstract parent class for filtering data"""
    def __init__(self):
        continue


def wavelet_filter(filter):
    """Implements wavelet filtering for ECEI data.

    Parameters:
    -----------
    kwargs....: dict, kwargs for call to denoise_wavelet
    """

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def __call__(self, data, inplace=False):
        """Applies wavelet_filter

        Parameters:
        -----------
        data......: ndarray, axis0: channels, axis1: time
        inplace...: bool, If True, replace data with the filtered data. If False, return a 
                    new ndarray with the filtered data
        """

        if not inplace:
            newdata = np.zeros_like(data)

        for ch_idx in range(data.shape[0]):
            signal = data[ch_idx, :]
            sigma = estimate_sigma(signal)

            if not inplace:
                newdata[ch_idx, :] = denoise_wavelet(signal, sigma=sigma, **self.kwargs)
            else:
                data[ch_idx, :] = denoise_wavelet(signal, sigma=sigma, **self.kwargs)


        if inplace:
            return None
        
        return newdata


# End of file wavelet_filters.py