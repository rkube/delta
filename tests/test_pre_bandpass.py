# -*- Encoding: UTF-8 -*-

"""Unit tests for wavelet preprocessing."""


def test_pre_bandpass_iir(config_all):
    """Compare Delta's bandpass preprocessing to directly calling scipy signal."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))

    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from data_models.kstar_ecei import ecei_chunk
    from preprocess.pre_bandpass import pre_bandpass_iir
    from azure.storage.blob import BlockBlobService

    from scipy.signal import iirdesign, sosfiltfilt

    # Create the BlockBlobService that is used to call the Blob service for the storage account
    blob_service_client = BlockBlobService(account_name="deltafiles")
    # Files are stored in testfiles
    container_name = 'testfiles'
    local_file_name = "ecei_22289_preprocess_data.npz"

    # Download the blob(s).
    # Add '_DOWNLOADED' as prefix to '.txt' so you can see both files in Documents.
    # shifter is read-only so let's use /tmp, the only place we're allowed to read/write
    full_path_to_file = os.path.join('/tmp',
                                     str.replace(local_file_name, '.npz', '_DOWNLOADED.npz'))
    blob_service_client.get_blob_to_path(container_name, local_file_name, full_path_to_file)

    with np.load(full_path_to_file) as df:
        data_orig = df["orig_data"]
        filt_fluctana = df["filt_data"]

    # Clean up the temp file
    os.remove(full_path_to_file)

    # Filter data here
    fsample = 5e5   # Sampling frequency,in Hz
    wp = config_all["preprocess"]["bandpass_iir"]["wp"]
    ws = config_all["preprocess"]["bandpass_iir"]["ws"]
    gpass = config_all["preprocess"]["bandpass_iir"]["gpass"]
    gstop = config_all["preprocess"]["bandpass_iir"]["gstop"]
    ftype = config_all["preprocess"]["bandpass_iir"]["ftype"]
    sos = iirdesign(wp, ws, gpass, gstop, ftype=ftype, output="sos")
    y_filt_here = sosfiltfilt(sos, data_orig, axis=1)
    y_filt_here[np.isnan(y_filt_here)] = 0.0

    # Filter data using Delta
    data_chunk = ecei_chunk(data_orig, None)
    my_pre_bandpass_iir = pre_bandpass_iir(config_all["preprocess"]["bandpass_iir"])
    e = ThreadPoolExecutor(max_workers=2)
    y_filt_delta = my_pre_bandpass_iir.process(data_chunk, e)
    y_filt_delta.data[np.isnan(y_filt_delta.data)] = 0.0


    # Test if data are the same
    assert(np.linalg.norm(y_filt_delta.data - y_filt_here) < 1e-10)

    # Assert that the 
    assert(np.linalg.norm(y_filt_delta.data[30, :] - filt_fluctana[30, :]) / 
           np.linalg.norm(y_filt_delta.data[30, :]) < 1.0)

# End of file test_pre_wavelet.py
