# -*- Encoding: UTF-8 -*-

"""Test coherence kernel."""


def test_kernel_coherence(gen_sine_waves):
    """Test coherence."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    from delta.analysis.kernels_spectral import kernel_coherence
    from delta.data_models.channels_2d import channel_2d, channel_pair

    ch1 = channel_2d(1, 1, 2, 1, "horizontal")
    ch2 = channel_2d(2, 1, 2, 1, "horizontal")
    ch_pair = channel_pair(ch1, ch2)

    fft_data = gen_sine_waves
    coherence = kernel_coherence(fft_data, [ch_pair], None)

    assert((np.abs(coherence.mean()) - 1.0) < 1e-8)


def test_kernel_crosspower(gen_sine_waves):
    """Test crosspower."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from delta.analysis.kernels_spectral import kernel_crosspower
    from delta.data_models.channels_2d import channel_2d, channel_pair

    ch1 = channel_2d(1, 1, 2, 1, "horizontal")
    ch2 = channel_2d(2, 1, 2, 1, "horizontal")
    ch_pair = channel_pair(ch1, ch2)

    fft_data = gen_sine_waves
    crosspower = kernel_crosspower(fft_data, [ch_pair], {"win_factor": 1})

    assert(crosspower[0, 2] > crosspower.mean())
    assert(crosspower[0, 8] > crosspower.mean())


def test_kernel_crossphase(gen_sine_waves):
    """Test crosspower."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    from delta.analysis.kernels_spectral import kernel_crossphase
    from delta.data_models.channels_2d import channel_2d, channel_pair

    ch1 = channel_2d(1, 1, 2, 1, "horizontal")
    ch2 = channel_2d(2, 1, 2, 1, "horizontal")
    ch_pair = channel_pair(ch1, ch2)

    fft_data = gen_sine_waves
    crossphase = kernel_crossphase(fft_data, [ch_pair], {"win_factor": 1})

    assert(np.abs(np.abs(crossphase[0, 2]) - 0.25) < 1e-7)
    assert(np.abs(np.abs(crossphase[0, 8]) - 0.5) < 1e-7)



# End of file test_kernel_coherence.py
