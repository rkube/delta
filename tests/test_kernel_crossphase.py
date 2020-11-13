# -*- Encoding: UTF-8 -*-

import pytest

"""Test cross-correlation kernel."""


@pytest.fixture
def gen_sine_waves():
    """Generate sine wave data for coherence kernel.

    Creates two signals with two frequencies.
    Each frequency has a distinct phase shift.

    See kstar_test_coherence.ipynb
    """
    import numpy as np
    from scipy.signal import stft

    # Number of realizations of the signal
    num_realizations = 5
    # Sampels per realization
    samples_per_realization = 100
    t0 = 0.0
    t1 = 1.0
    dt = 1e-2
    # Time range of a single realization
    trg = np.arange(t0, t1 * num_realizations, dt)

    # Pre-allocate wave data array
    wave_data = np.zeros([2, num_realizations * samples_per_realization])

    # Base frequencies and phase shift of each frequency
    f0 = 1.0
    f1 = 4.0
    delta_phi_f0 = 0.25 * (t1 - t0)
    delta_phi_f1 = 0.5 * (t1 - t0)

    # Calculate y
    wave_data[0, :] = np.sin(f0 * 2.0 * np.pi * trg) + np.sin(f1 * 2.0 * np.pi * trg)
    wave_data[1, :] = np.sin(f0 * 2.0 * np.pi * (trg - delta_phi_f0)) + np.sin(f1 * 2.0 * np.pi * (trg - delta_phi_f1))

    # Pre-allocate FFT data array.
    num_bins = 11
    nfft = samples_per_realization // 2 + 1
    fft_data = np.zeros([2, nfft, num_bins], dtype=np.complex128)
    for ch in [0, 1]:
        f_s = stft(wave_data[ch, :], nperseg=100)
        fft_data[ch, :, :] = f_s[2][:, ]

    return fft_data


def test_kernel_crosscorr(caplog, config_all, gen_sine_waves):
    """Test cross-correlation."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    from delta.analysis.kernels_spectral import kernel_coherence
    from delta.data_models.channels_2d import channel_2d, channel_pair

    import logging
    logger = logging.getLogger(__name__)
    caplog.set_level(logging.INFO)

    ch1 = channel_2d(1, 1, 2, 1, "horizontal")
    ch2 = channel_2d(2, 1, 2, 1, "horizontal")
    ch_pair = channel_pair(ch1, ch2)

    logger.info(f"{ch1.get_idx()} {ch2.get_idx()}")

    fft_data = gen_sine_waves
    coherence = kernel_coherence(fft_data, [ch_pair], None)

    assert((np.abs(coherence.mean()) - 1.0) < 1e-8)


# End of file test_kernel_coherence.py
