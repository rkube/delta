# -*- Encoding: UTF-8 -*-

"""Unit tests for pure python kernels."""


def test_kernels(config_all):
    """Verify that pure python kernels yield same result as fluctana."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))

    import numpy as np

    from concurrent.futures import ThreadPoolExecutor
    from analysis.task_list import tasklist
    from preprocess.preprocess import preprocessor
    from data_models.kstar_ecei import ecei_chunk
    from data_models.helpers import get_dispatch_sequence
    from data_models.timebase import timebase_streaming
    from analysis.kernels_spectral import kernel_crossphase

    from scipy.signal import stft

    from azure.storage.blob import BlockBlobService

    # Mutate storage path for numpy backend
    config_all["storage"]["basedir"] = os.getcwd()

    # Create the BlockBlobService that is used to call the Blob service for the storage account
    blob_service_client = BlockBlobService(account_name="deltafiles")
    # Files are stored in testfiles
    container_name = 'testfiles'
    local_file_name = "ecei_22289_analyzed_fluctana.npz"

    # Download the blob(s).
    # Add '_DOWNLOADED' as prefix to '.txt' so you can see both files in Documents.
    full_path_to_file = os.path.join(os.getcwd(), 
                                     str.replace(local_file_name, '.npz', '_DOWNLOADED.npz'))
    blob_service_client.get_blob_to_path(container_name, local_file_name, full_path_to_file)

    # Load relevant data into local scope
    with np.load(full_path_to_file) as df:
        sig_ch0_fa = np.squeeze(df["data_dlist0"])
        sig_ch1_fa = np.squeeze(df["data_dlist1"])
        t_start, t_end = df["trange"]
    # flimits is also in the numpy file, but it was stored as a string...
    flimits = np.array([5, 9])
    os.remove(full_path_to_file)

    # Generate data. Load only 2 relevant channel data into the dummy data block.
    # This data has been frequency filtered in fluctana.
    all_data = np.zeros([192, 10_000])
    ch0_idx = 1 * 8 + 6 - 1   # clist1 = 0206
    ch1_idx = 1 * 8 + 5 - 1   # clist1 = 0205
    all_data[ch0_idx, :] = sig_ch0_fa[:]
    all_data[ch1_idx, :] = sig_ch1_fa[:]
    # Calculate cross-phase from fluctana
    frg, _, sig_ch0_bp_ft = stft(sig_ch0_fa, fs=5e5, nperseg=config_all["preprocess"]["stft"]["nfft"], noverlap=256, detrend="constant", window="hann", return_onesided=False)
    frg, _, sig_ch1_bp_ft = stft(sig_ch1_fa, fs=5e5, nperseg=config_all["preprocess"]["stft"]["nfft"], noverlap=256, detrend="constant", window="hann", return_onesided=False)

    frg = np.fft.fftshift(frg)
    sig_ch0_bp_ft = np.fft.fftshift(sig_ch0_bp_ft, axes=0)
    sig_ch1_bp_ft = np.fft.fftshift(sig_ch1_bp_ft, axes=0)

    # Compute crossphase from fluctana
    Pxy_bp = sig_ch0_bp_ft * sig_ch1_bp_ft.conj()
    Axy_bp = np.arctan2(Pxy_bp.imag, Pxy_bp.real).real.mean(axis=1)

    # Set up Delta stuff
    e = ThreadPoolExecutor(max_workers=2)
    my_preprocess = preprocessor(e, config_all)
    my_tasklist = tasklist(e, config_all)
    tb = timebase_streaming(t_start, t_end, 5e5, 10000, 0)
    ecei_params = {"TFcurrent": 18000, "LensFocus": 503, "LoFreq": 79.5, "Mode": "X", "LensZoom": 200, "dev": "GT"}
    my_chunk = ecei_chunk(all_data, tb, ecei_params)
    # Pre-process chunk
    chunk_pre = my_preprocess.submit(my_chunk)
    # Calculate diagnostics using Delta
    my_tasklist.execute(chunk_pre)

    # Shutdown the executor, wait for all calculations to finish.
    e.shutdown(wait=True)

    # Load data from numpy file:
    fname_fq = os.path.join(os.getcwd(), "task_crossphase" + f"_chunk{0:05d}_batch{0:02d}.npz")
    with np.load(fname_fq) as df:
        crossphase_delta = df["arr_0"]

    # Get the index where the relevant data is stored
    # This is the dispatch sequence for KSTAR ECEi
    seq = get_dispatch_sequence([1, 1, 24, 8], [1, 1, 24, 8], niter=20000)
    ch_it = seq[0]
    # Iterate over the channel pairs and capture the index where ch1/ch0 is calculated in Delta
    delta_idx = -1
    for idx, ch_pair in enumerate(ch_it):
        if(ch_pair.ch1.get_idx() == ch1_idx):
            if(ch_pair.ch2.get_idx() == ch0_idx):
                print(ch_pair.ch1.get_idx(), ch_pair.ch2.get_idx())
                delta_idx = idx
                break

    # Alternative 2: Pass pre-processed chunk data directly to kernel
    Axy_here = kernel_crossphase(chunk_pre.data_ft, ch_it, None)

    # Calcuate the L2 norm between the crossphase calculated here to the crossphase calculated in Delta
    # Do this only for the frequencies within the filter pass band
    cmp_idx = (np.abs(frg) > flimits[0] * 1e3) & (np.abs(frg) < flimits[1] * 1e3)
    dist = np.linalg.norm(Axy_bp[cmp_idx] + 
                          crossphase_delta[delta_idx, cmp_idx]) / np.linalg.norm(Axy_bp[cmp_idx])

    assert(dist < 0.5)

    # Calculate the L2 norm between crossphase calcuated through task_list.submit and
    # by directly calling the kernel
    dist = np.linalg.norm(Axy_here - crossphase_delta)
    assert(dist < 1e-10)    





    # Clean up the temp file
    



# End of file test_kernels_py.py