# -*- Encoding: UTF-8 -*-

"""Unit tests for preprocess/helpers.py."""

def test_preprocess_helpers(config_all):
    """Tests channel_2d object."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from preprocess.helpers import get_preprocess_routine
    from preprocess.pre_stft import pre_stft
    from preprocess.pre_bandpass import pre_bandpass_iir

    type_list = [pre_bandpass_iir, pre_stft]

    for key, params, mytype in zip(config_all["preprocess"].keys(),
                                   config_all["preprocess"].items(),
                                   type_list):
        vv = get_preprocess_routine(key, params[1])
        assert(isinstance(vv, mytype))



# End of file test_preprocess_helpers.py
