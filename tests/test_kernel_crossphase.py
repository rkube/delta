# -*- Encoding: UTF-8 -*-

"""Test coherence kernel."""


def test_kernel_coherence(caplog, gen_sine_waves):
    """Test coherence."""
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
