# -*- Encoding: UTF-8 -*-

"""
Author: Ralph Kube

Contains helper functions for pre-processing setup
"""
from preprocess.pre_stft import pre_stft
from preprocess.pre_bandpass import pre_bandpass
from preprocess.pre_wavelet import pre_wavelet


def get_preprocess_routine(key, params):
    """Returns the appropriate pre-processing routine. 
       Raises NameError if the key can not be matched
       to an available preprocessing routine."""

    if key == "stft":
        return pre_stft(params)
    elif key == "wavelet":
        return pre_wavelet(params)
    elif key == "bandpass":
        return pre_bandpass(params)
    else:
        raise NameError(f"Requested invalid pre-processing routine: {key}")




# End of file helpers.py