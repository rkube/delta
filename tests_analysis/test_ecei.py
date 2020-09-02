# -*- Encoding: UTF-8 -*-

from analysis.ecei_channel import timebase_streaming


# Test if the streaming time-base works correctly

t_start = -0.1
t_end = 9.9
f_sample = 5e5
chunk_size = 10_000

tb_stream = timebase_streaming(t_start, t_end, f_sample, chunk_size, 12)

# Case 1) See if the indices are mapped correctly when transitioning from
# chunk 11 to chunk 12

for t, target in zip(range(12 * chunk_size - 2, 12 * chunk_size + 2),
                     [None, None, 0, 1, 2]):
    time = t_start + t / f_sample
    assert tb.time_to_idx(time) == target


# End of file test_ecei.py