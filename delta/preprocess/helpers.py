# -*- Encoding: UTF-8 -*-

"""Contains helper functions for pre-processing pipeline."""


from preprocess.pre_stft import pre_stft
from preprocess.pre_bandpass import pre_bandpass_fir, pre_bandpass_iir
from preprocess.pre_wavelet import pre_wavelet
from preprocess.pre_plot import pre_plot


def get_preprocess_routine(key, params):
    """Returns an instance of a pre-processing objects that matches the key.

    +--------------+------------------------------------------------------+
    | key          | object                                               |
    +==============+======================================================+
    | stft         | :py:class:`preprocess.pre_stft.pre_stft`             |
    +--------------+------------------------------------------------------+
    | wavelet      | :py:class:`preprocess.pre_wavelet.pre_wavelet`       |
    +--------------+------------------------------------------------------+
    | bandpass_fir | :py:class:`preprocess.pre_bandpass.pre_bandpass_fir` |
    +--------------+------------------------------------------------------+
    | bandbass_iir | :py:class:`preprocess.pre_bandpass.pre_bandpass_iir` |
    +--------------+------------------------------------------------------+
    | plot         | :py:class:`preprocess.pre_plot.pre_plot`             |
    +--------------+------------------------------------------------------+


    Args:
        key (string):
            Name of the pre-processing routine
        params (dictionary):
            Dictionary containig kwargs for pre-processing routines.

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
    elif key == "plot":
        return pre_plot(params)
    else:
        raise NameError(f"Requested invalid pre-processing routine: {key}")


# End of file helpers.py
