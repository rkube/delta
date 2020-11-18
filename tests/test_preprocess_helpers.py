# -*- Encoding: UTF-8 -*-

"""Unit tests for preprocess/helpers.py."""

def test_preprocess_helpers(config_all):
    """Tests channel_2d object."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from delta.preprocess.helpers import get_preprocess_routine
    from delta.preprocess.pre_stft import pre_stft
    from delta.preprocess.pre_wavelet import pre_wavelet
    from delta.preprocess.pre_plot import pre_plot

    type_list = [pre_plot, pre_wavelet, pre_stft]

    for key, params, mytype in zip(config_all["preprocess"].keys(),
                                   config_all["preprocess"].items(),
                                   type_list):
        vv = get_preprocess_routine(key, params, config_all["diagnostic"])
        assert(type(vv) == mytype)



# End of file test_preprocess_helpers.py
