# -*- Encoding: UTF-8 -*-

"""Unit tests for channels_2d.py."""

def test_channel_2d(config_all):
    """Tests channel_2d object."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from delta.data_models.channels_2d import channel_2d

    ch_v = 6
    ch_h = 3
    ch = channel_2d(ch_v, ch_h, 24, 8, "horizontal")

    assert(ch.__str__() == f"({ch_v:03d}, {ch_h:03d})")
    assert(ch == channel_2d(ch_v, ch_h, 24, 8, "horizontal"))
    assert(ch.get_num() == (ch_v - 1) * 8 + ch_h)
    assert(ch.get_idx() == (ch_v - 1) * 8 + ch_h - 1)


def test_channel_range_h(config_all):
    """Tests channel_range object."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from delta.data_models.channels_2d import channel_2d, channel_range

    ch_start = channel_2d(2, 3, 6, 6, "horizontal")
    ch_end = channel_2d(5, 5, 6, 6, "horizontal")

    ch_rg = channel_range(ch_start, ch_end)
    assert(ch_rg.length() == 12)
    i = 0
    for _ in ch_rg:
        i += 1
    assert(i == ch_rg.length())


def test_channel_range_v(config_all):
    """Tests channel_range object."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from delta.data_models.channels_2d import channel_2d, channel_range

    ch_start = channel_2d(2, 3, 6, 6, "vertical")
    ch_end = channel_2d(5, 5, 6, 6, "vertical")

    ch_rg = channel_range(ch_start, ch_end)
    assert(ch_rg.length() == 12)
    i = 0
    for _ in ch_rg:
        i += 1
    assert(i == ch_rg.length())


def test_channel_pair(config_all):
    """Tests member functions of channel_pair."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    from delta.data_models.channels_2d import channel_2d, channel_pair

    ch1 = channel_2d(13, 7, 24, 8, 'horizontal')
    ch2 = channel_2d(12, 7, 24, 8, 'horizontal')

    assert(channel_pair(ch1, ch2) == channel_pair(ch2, ch1))


# End of file test_channels_2d.py
