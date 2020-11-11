# -*- Encoding: UTF-8 -*-

"""Unit tests for data normalization."""

import pytest

@pytest.fixture
def gen_dummy_data():
    """Generates dummy data as a place-holder for ECEI data.

    Dummy data is a sine wave plus additional noise on each channel.
    """
    import numpy as np

    n_sample = 10_000
    n_ch = 192
    scale_noise = 0.05
    offset = 1.0

    n_waves = np.random.randint(5, 10, n_ch)
    phase = np.random.uniform(-0.5, 0.5, n_ch)

    trg = np.linspace(0.0, 1.0, n_sample, endpoint=False)

    data_arr = np.sin(2. * np.pi * n_waves.repeat(n_sample).reshape(n_ch, n_sample) *
                    (trg.reshape(1, n_sample).repeat(n_ch, axis=0) +

                    phase.repeat(n_sample).reshape(n_ch, n_sample))) +\
        trg.reshape(1, n_sample).repeat(n_ch, axis=0)

    noise = np.random.normal(loc=0.0, scale=scale_noise, size=data_arr.shape)
    return data_arr + noise + offset


def test_normalization(gen_dummy_data, caplog):
    """Verify normaliation procedure."""
    import numpy as np


    from data_models.kstar_ecei import ecei_chunk
    from data_models.helpers import normalize_mean
    #dummy_data = gen_dummy_data

    my_chunk = ecei_chunk(gen_dummy_data, tb=None)
    norm = normalize_mean(my_chunk.data)
    norm(my_chunk)

    # offlev should roughly be 1, but not very close.
    assert(np.abs(norm.offlev.mean() - 1.5) < 1e-2)
    # offstd should be roughly 0.76
    assert(np.abs(norm.offstd.mean() - 0.76) < 1e-2)
    # Normalized data should have rioughly zero mean
    assert(np.abs(my_chunk.data.mean()) < 1e-8)






# # End of file test_normalization_kstar_ecei.py
