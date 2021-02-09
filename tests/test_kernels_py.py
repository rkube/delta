# -*- Encoding: UTF-8 -*-

"""Unit tests for pure python kernels."""

from unittest.mock import patch



#@patch("module.adios2")
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
    from analysis.kernels_spectral import kernel_crossphase, kernel_coherence, kernel_crosspower


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
        # "ground truth": fluctana results
        crossphase_fa = np.squeeze(df["cross_phase_data"])
        crosspower_fa = np.squeeze(df["cross_power_data"]) 
        coherence_fa = np.squeeze(df["coherence_data"])

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
    nfft = config_all["preprocess"]["stft"]["nfft"]
    frg, _, sig_ch0_bp_ft = stft(sig_ch0_fa, fs=5e5, nperseg=nfft, noverlap=256, detrend="constant", window="hann", return_onesided=False)
    frg, _, sig_ch1_bp_ft = stft(sig_ch1_fa, fs=5e5, nperseg=nfft, noverlap=256, detrend="constant", window="hann", return_onesided=False)

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
    dist = np.linalg.norm(Axy_here - crossphase_delta) / np.linalg.norm(Axy_here)
    assert(dist < 0.1)    

    ############################# Testing cross-power #############################################
    #  Option 1) Load analysis result from Delta numpy file
    fname_fq = os.path.join(os.getcwd(), "task_crosspower" + f"_chunk{0:05d}_batch{0:02d}.npz")
    with np.load(fname_fq) as df:
        crosspower_delta = df["arr_0"]

    # Option 2) Load fluctana bandpass filtered data from file and calculate manually
    win_factor = np.mean(np.hanning(nfft)**2.0)
    Pxy_bp = (sig_ch0_bp_ft * sig_ch1_bp_ft.conj()).real.mean(axis=1) / win_factor
    
    # Option 3) Pass bandpass-filtered data into python kernel
    params = {"win_factor": win_factor}
    Pxy_kernel = kernel_crosspower(chunk_pre.data_ft, ch_it, params)

    # Dist from 1 and 2 should be small
    dist = np.linalg.norm(np.log10(Pxy_bp[cmp_idx] / 
                                   crosspower_delta[delta_idx, cmp_idx])) / np.linalg.norm(np.log10(Pxy_bp[cmp_idx]))
    assert(dist < 0.1)

    # Dist from 2 to 3 should be zero-ish
    dist = np.linalg.norm(Pxy_kernel - crosspower_delta) / np.linalg.norm(Pxy_kernel)
    assert(dist < 0.1)

    # Dist from 1 to original fluctana should not be too large
    dist = np.linalg.norm(np.log10(crosspower_fa[:-1][cmp_idx] / crosspower_delta[delta_idx, cmp_idx])) / np.linalg.norm(np.log10(crosspower_delta[delta_idx, cmp_idx]))
    assert(dist < 0.1)


    ############################# Testing coherence #############################################
    # Option 1) Load analysis result from Delta numpy file
    fname_fq = os.path.join(os.getcwd(), "task_coherence" + f"_chunk{0:05d}_batch{0:02d}.npz")
    with np.load(fname_fq) as df:
        coherence_delta = df["arr_0"]

    # Option 2) Load fluctana bandpass-filtered signal and calculate coherence manually
    Pxx = sig_ch0_bp_ft * sig_ch0_bp_ft.conj()
    Pyy = sig_ch1_bp_ft * sig_ch1_bp_ft.conj()
    Gxy_bp = np.abs((sig_ch0_bp_ft * sig_ch1_bp_ft.conj() / np.sqrt(Pxx * Pyy)).mean(axis=1)).real

    # Option 3) Pass pre-processed data directly into python kernel
    Gxy_kernel = kernel_coherence(chunk_pre.data_ft, ch_it, None)

    print("coherence_delta = ", coherence_delta[delta_idx, cmp_idx])
    print("Gxy_bp = ", Gxy_bp[cmp_idx])
    print("Gxy_kernel = ", Gxy_kernel[delta_idx, cmp_idx])

    np.savez("coherence_cpu.npz", Gxy_kernel=Gxy_kernel)

    # Dist from 1 and 2 should be small
    dist = np.linalg.norm(Gxy_bp[cmp_idx] -
                          coherence_delta[delta_idx, cmp_idx]) / np.linalg.norm(Gxy_bp[cmp_idx])
    assert(dist < 0.5)

    # Dist from 2 to 3 should be zero-ish
    dist = np.linalg.norm(Gxy_kernel[delta_idx, cmp_idx] - coherence_delta[delta_idx, cmp_idx]) / np.linalg.norm(Gxy_kernel[delta_idx, cmp_idx])
    assert(dist < 0.1)

    # Dist to fluctana should not be too ugly
    dist = np.linalg.norm(coherence_fa[:-1][cmp_idx] - coherence_delta[delta_idx, cmp_idx], ord=2) / np.linalg.norm(coherence_delta[delta_idx, cmp_idx])
    assert(dist < 0.5)




# End of file test_kernels_py.py