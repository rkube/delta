# -*- Encoding: UTF-8 -*-

"""
Contains helper functions for pre-processing setup
"""
from preprocess.pre_stft import pre_stft
from preprocess.pre_bandpass import pre_bandpass_fir, pre_bandpass_iir
from preprocess.pre_wavelet import pre_wavelet


def get_preprocess_routine(key, params):
    """Returns a instance of a callable pre-processing objects for each key.

        Args:
            key (string):
                Name of the pre-processing routine
            params (dictionary):
                Parameters passed to the pre-processing object

        Returns:
            obj (callable):
                Callable pre-processing object

        Raises:
            NameError:
                If the key can not be matched to an available pre-processing object.

    """

    if key == "stft":
        return pre_stft(params)
    elif key == "wavelet":
        return pre_wavelet(params)
    elif key == "bandpass_fir":
        return pre_bandpass_fir(params)
    elif key == "bandpass_iir":
        return pre_bandpass_iir(params)
    else:
        raise NameError(f"Requested invalid pre-processing routine: {key}")


# End of file helpers.py
