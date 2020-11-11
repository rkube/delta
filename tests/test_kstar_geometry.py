#-*- Encoding: UTF-8 -*-

"""Verfify calculated ECEI channel positions against fluctana."""


def test_ecei_channel_geom():
    """Verify calculated ECEI channel positions are the same as in fluctana."""
    import sys
    import numpy as np
    from data_models.channels_2d import channel_2d
    from data_models.kstar_ecei import get_geometry
    # Add fluctana repository to calculate ground truth
    sys.path.append("/global/homes/r/rkube/repos/fluctana_rmc")
    from kstarecei import KstarEcei

    # Pick a random channel
    ch_h = np.random.randint(1, 9)
    ch_v = np.random.randint(1, 25)
    ch_str = f"ECEI_L{ch_v:02d}{ch_h:02d}"
    ch_2d = channel_2d(ch_v, ch_h, 24, 8, order="horizontal")

    # Manually provide ECEI configuration dictionary for Delta
    cfg_diagnostic = {"name": "kstarecei", "shotnr": 18431,
                      "datasource": {
                          "source_file": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/018431/ECEI.018431.LFS.h5",
                          "chunk_size": 10000,
                          "num_chunks": 500,
                          "channel_range": ["L0101-2408"],
                          "datatype": "float"},
                      "parameters": {
                          "Device": "L",
                          "TriggerTime": [-0.12, 61.2, 60],
                          "t_norm": [-0.119, -0.109],
                          "SampleRate": 500,
                          "TFcurrent": 23000.0,
                          "Mode": "O",
                          "LoFreq": 81,
                          "LensFocus": 80,
                          "LensZoom": 340}}

    # Calcuate channel position using FluctAna and Delta
    K = KstarEcei(shot=18431, clist=[ch_str], data_path='/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/')
    K.channel_position()
    pos_true = np.array([K.rpos[0], K.zpos[0], K.apos[0]])
    rpos_arr, zpos_arr, apos_arr = get_geometry(cfg_diagnostic["parameters"])
    # print("Re-factored:", ch_2d.get_idx())
    # print(f"rpos = {rpos_arr[ch_2d.get_idx()]}, zpos = {zpos_arr[ch_2d.get_idx()]}, apos = {apos_arr[ch_2d.get_idx()]}")
    pos_delta = np.array([rpos_arr[ch_2d.get_idx()], zpos_arr[ch_2d.get_idx()], apos_arr[ch_2d.get_idx()]])

    assert(np.linalg.norm(pos_true - pos_delta) < 1e-8)


# End of file test_kstar_ecei_helpers.py
