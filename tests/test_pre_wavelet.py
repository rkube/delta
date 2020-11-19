# -*- Encoding: UTF-8 -*-

"""Unit tests for wavelet preprocessing."""

import pytest

def test_pre_wavelet(config_all):
    """Verify that wavelet pre-processing works fine."""
    import sys
    import os
    from google_drive_downloader import GoogleDriveDownloader as gdd
    sys.path.append(os.path.abspath('delta'))

    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from data_models.kstar_ecei import ecei_chunk
    from preprocess.pre_wavelet import pre_wavelet

    
    gdd.download_file_from_google_drive(file_id='1TDfi-gLLphzXsp4BGoiWUFVT8Kt3KXPp', 
                                        dest_path="./test_pre_wavelet_data.npz")
    with np.load("test_pre_wavelet_data.npz") as df:
        ch_data2 = df["ch_data2"]

    data_chunk = ecei_chunk(ch_data2.reshape(192, 1000), None)
    my_pre_wavelet = pre_wavelet(config_all["preprocess"]["wavelet"])
    e = ThreadPoolExecutor(max_workers=2)

    _ = my_pre_wavelet.process(data_chunk, e)
    # This test just needs to execute until here. We don't verify the actual result of
    # the preprocessing 
    assert(True)


# End of file test_pre_wavelet.py
