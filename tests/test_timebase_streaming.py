# -*- Encoding: UTF-8 -*-

"""Unit tests for data_models.timebase_streaming.py."""

import pytest


@pytest.fixture
def init_timebase():
    """Initializes a timebase_streaming object.

    See introduction to fixtures: https://docs.pytest.org/en/stable/fixture.html#fixtures
    """
    from data_models.timebase import timebase_streaming
    # Time-range starts at -0.1s and goes until 9.9s
    # Sampling frequency is 5e5 Hz -> dt = 2e-6
    # 10_000 samples per chunk.
    t_start = -0.1
    t_end = 9.9
    f_sample = 5e5
    chunk_size = 10_000

    # Create a streaming time-base for chunk_idx=12
    # In chunk_idx=12 we have samples from,
    # t_0 = t_start + 12 * 10_000 * dt = 0.14s
    # t_1 = t_start + (13 * 10_000 - 1) * dt = 0.159998s
    chunk_idx = 12
    tb_stream = timebase_streaming(t_start, t_end, f_sample, chunk_size, chunk_idx)

    return tb_stream


def test_ttidx(init_timebase):
    """Test if the timebase_streaming correctly maps indices to time values."""
    tb_stream = init_timebase
    chunk_idx = tb_stream.chunk_idx
    chunk_size = tb_stream.samples_per_chunk

    # Case 1) See if the indices are mapped correctly when transitioning from
    # chunk 11 to chunk 12
    for t, target in zip(range(chunk_idx * chunk_size - 2, chunk_idx * chunk_size + 2),
                         [None, None, 0, 1, 2]):
        time = tb_stream.t_start + t / tb_stream.f_sample
        assert tb_stream.time_to_idx(time) == target


def test_t0t1(init_timebase):
    """Test if timebase_streaming captures the correct time range."""
    tb_stream = init_timebase
    t0, t1 = tb_stream.get_trange()
    assert(abs(t0 - 0.14) < 1e-10)
    assert(abs(t1 - 0.159998) < 1e-10)

# End of file test_timebase_streaming.py