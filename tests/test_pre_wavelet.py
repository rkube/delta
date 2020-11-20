# -*- Encoding: UTF-8 -*-

"""Unit tests for wavelet preprocessing."""

import pytest

def test_pre_wavelet(config_all):
    """Verify that wavelet pre-processing works fine."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))

    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from data_models.kstar_ecei import ecei_chunk
    from preprocess.pre_wavelet import pre_wavelet
    from azure.storage.blob import BlockBlobService

    # Create the BlockBlobService that is used to call the Blob service for the storage account
    blob_service_client = BlockBlobService(account_name="deltafiles")
    # Files are stored in testfiles
    container_name = 'testfiles'
    local_file_name = "ch_data2.npz"

    # Download the blob(s).
    # Add '_DOWNLOADED' as prefix to '.txt' so you can see both files in Documents.
    full_path_to_file2 = os.path.join(os.getcwd(), str.replace(local_file_name, '.npz', '_DOWNLOADED.npz'))
    blob_service_client.get_blob_to_path(container_name, local_file_name, full_path_to_file2)

    with np.load("ch_data2_DOWNLOADED.npz") as df:
        ch_data2 = df["ch_data2"]

    # Clean up the temp file
    os.remove(full_path_to_file2)

    data_chunk = ecei_chunk(ch_data2.reshape(192, 1000), None)
    my_pre_wavelet = pre_wavelet(config_all["preprocess"]["wavelet"])
    e = ThreadPoolExecutor(max_workers=2)

    _ = my_pre_wavelet.process(data_chunk, e)
    # This test just needs to execute until here. We don't verify the actual result of
    # the preprocessing 
    assert(True)


# End of file test_pre_wavelet.py
