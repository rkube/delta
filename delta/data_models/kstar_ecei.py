# -*- Encoding: UTF-8 -*-

"""Defines data types and helper functions for handling KSTAR ECEI data.

The ECEI diagnostics provides 192 independent channels that are arranged
into a 8 radial by 24 vertical view.

This code is based on the `fluctana <https://github.com/minjunJchoi/fluctana>`_ code
"""

import logging
import numpy as np

from data_models.base_models import twod_chunk
from data_models.channels_2d import channel_2d, channel_range, num_to_vh


class ecei_chunk(twod_chunk):
    """Class that represents a time-chunk of ECEI data.

    This class provides the following interface.

    Creating a time-chunk from streaming data, where `tb_chunk` is of type
    :py:class:`data_models.timebase.timebase_streaming`:

    .. code-block:: python

        chunk = ecei_chunk(stream_data, tb_chunk, stream_attrs)

    Example: Fourier-transformation of a time-chunk. Time-axis is given given by
    axis_t member, data is access by data member, sampling frequency is calculated
    through time-base.

    .. code-block:: python

        fft_data = fft(chunk.data, axis=chunk.axsi_t, fsample = 1. / chunk.tb.dt()

    """

    def __init__(self, data, tb, params=None, num_v=24, num_h=8):
        """Creates an ecei_chunk from a give dataset.

        The first dimension indices channels, the second dimension indices time.
        Channels are ordered.

        Parameters under which the data was measured needs to be passed in the params dict.
        Keys need to include:

            * dev: String identifier of the ECEI device: L, G, H, GT, or GR
            * TriggerTime:
            * t_norm: Vector of 2 floats, defining the time interval used for normalization. In seconds.
            * SampleRate: Rate at which each channels samples the plasma. In Hz.
            * TFCurrent: Toroidal Field Coil current, in Amps
            * Mode: string, either 'X' or 'O'
            * LoFreq: float
            * LensFocus: float
            * LensZoom: float

        Args:
            data (ndarray, float):
                Raw data for the ECEI voltages
            tb (:py:class:`data_models.timebase.timebase_streaming`):
                timebase for ECEI voltages
            params:
                Additional parameters under which the data was measured. See dicussion above.
            num_v (int):
                Number of vertical channels. Defaults to 24.
            num_h (int):
                Number of horizontal channels. Defaults to 8.

        Returns:
            None
        """
        
        super().__init__(data)  # Initializes data member
        self.logger = logging.getLogger("simple")
        # Data should have more than 1 dimension, last dimension is time
        assert(data.ndim > 1)
        #
        self.num_v, self.num_h = num_v, num_h

        # # We should ensure that the data is contiguous so that we can remove this from
        # # if not data.flags.contiguous:
        # self.ecei_data = np.require(data, dtype=np.float64, requirements=['C', 'O', 'W', 'A'])
        # assert(self.ecei_data.flags.contiguous)

        # Time-base for the chunk
        self.tb = tb
        # Axis that indexes channels
        self.axis_ch = 0
        # Axis that indexes time
        self.axis_t = 1
        # Data can be 2 or 3 dimensional
        assert(data.shape[self.axis_ch] == self.num_h * self.num_v)

        # Parameters for the ECEI chunk:
        self.params = params
        # True if data is normalized, False if not.
        self.is_normalized = False
        # offlev, offstd, siglev and sigstd have shape=(nchannels, 1)
        # This allows for broadcase operations
        self.offlev = None
        self.offstd = None
        self.siglev = None
        self.sigstd = None
        # bad_channels is used as a mask and has shape=(nchannels)
        self.bad_channels = np.zeros((self.num_h * self.num_v), dtype=bool)


    def mark_bad_channels(self, verbose=False):
        """Mark bad channels.

        A channel where any of the following three condition is true is marked as bad.

            * Low signal level: std(offset) / siglev > 0.3
            * Saturated signal data(bottom saturation): std(offset) < 0.001
            * Saturated offset data(top saturation): std(signal) < 0.001

        Internally, bad channels is represented by an bool array of shape (self.num_h * self.num_v)
        """
        self.logger.info(f"offlev: {self.offlev.shape}, offstd: {self.offstd.shape}, siglev: {self.siglev.shape}, sigstd: {self.sigstd.shape}")
        # Check for low signal level
        ref = 100. * self.offstd / self.siglev
        ref[self.siglev < 0.01] = 100
        # Squeeze singleton dimensions so that we can do indexing with ref
        ref = np.squeeze(ref)

        my_num_to_vh = num_to_vh(24, 8, "horizontal")

        if verbose:
            for item in np.argwhere(ref > 30.0):
                self.logger.debug(f"LOW SIGNAL: channel({my_num_to_vh(item[0] + 1)}) ref = {ref[item[0]]:f}")
        self.bad_channels[ref > 30.0] = True

        # Mark bottom saturated channels
        self.bad_channels[np.squeeze(self.offstd < 1e-3)] = True
        if verbose:
            for item in np.argwhere(self.offstd < 1e-3):
                os = self.offstd[tuple(item)]
                ol = self.offlev[tuple(item)]
                self.logger.debug(f"SAT offset channel {my_num_to_vh(item[0] + 1)}: offstd = {os} offlevel = {ol}")

        # Mark top saturated channels
        self.bad_channels[np.squeeze(self.sigstd < 1e-3)] = True
        if verbose:
            for item in np.argwhere(self.sigstd < 1e-3):
                os = self.offstd[tuple(item)]
                ol = self.offlev[tuple(item)]
                self.logger.debug(f"SAT signal channel {my_num_to_vh(item[0] + 1)}: offstd = {os} offlevel = {ol}")

    def create_ft(self, fft_data, params):
        """Returns a fourier-transformed object.

        Args:
            fft_data (ndarray):
                Numerical data
            params (dict):
                Data passed to STFT function

        Returns:
            ecei_chunk_ft (ecei_chunk_ft):
                Chunk of Fourier-transformed data
        """
        return ecei_chunk_ft(fft_data, tb=self.tb,
                             freqs=None, params=params)


class ecei_chunk_ft(twod_chunk):
    """Represents a fourier-transformed time-chunk of ECEI data."""

    def __init__(self, data, tb, freqs, params=None, axis_ch=0, axis_t=1, num_v=24, num_h=8):
        """Initializes with data and meta-information.

        Args:
            data (ndarray, float):
                Fourier Coefficients
            tb (timebase streaming):
                Timebase of the original data.
            freqs (ndarray, float):
                Frequency vector
            params (dictionary):
                Parameters used to calculate the STFT
            axis_ch (int):
                axis which indices the channels
            axis_t (int):
                axis which indices the fourier frequencies
            num_v (int):
                Number of vertical channels
            num_h (int):
                Number of horizontal channels

        Returns:
            None
        """
        super().__init__(data)  # Initializes data
        self.tb = tb
        self.freqs = freqs
        self.params = params
        self.axis_ch = axis_ch
        self.axis_t = axis_t
        self.num_v = num_v
        self.num_h = num_h



def channel_range_from_str(range_str):
    """Generates a channel_range from a range.

    In `fluctana <https://github.com/minjunJchoi/fluctana>`_ ,
    channels are referred to f.ex.

    .. code-block::

        'L0101' or 'GT1507'

    The letters refer to a device (D), the first two digits to the vertical channel number (V)
    and the last two digits refer to the horizontal channel number (H). Delta uses the same DDVVHH
    format.

    Args:
        range_str (str):
            KSTAR ECEI channel range, format DDVVHH

    Returns:
        channel_range (:py:class:`data_models.channels_2d.channel_range`):
             Channel range corresponding to range_str
    """
    import re

    m = re.search('[A-Z]{1,2}', range_str)
    # try:
    #     dev = m.group(0)
    # except:
    #     raise AttributeError("Could not parse channel string " + range_str)

    m = re.findall('[0-9]{4}', range_str)

    ch_i = int(m[0])
    ch_hi = ch_i % 100
    ch_vi = int(ch_i // 100)
    ch_i = channel_2d(ch_vi, ch_hi, 24, 8, 'horizontal')

    ch_f = int(m[1])
    ch_hf = ch_f % 100
    ch_vf = int(ch_f // 100)
    ch_f = channel_2d(ch_vf, ch_hf, 24, 8, 'horizontal')

    return channel_range(ch_i, ch_f)


def get_abcd(LensFocus, LensZoom, Rinit, dev, new_H=True):
    """Returns ABCD matrix for KSTAR ECEI diagnostic.

    Args:
        LensZoom (float):
            LensZoom
        LensFocus (float):
            LensFocus
        Rinit (float):
            Radial position of the channel, in meter
        dev (char):
            Name ECEI device. Either one of 'L', 'H', 'G', 'GT', 'GR', 'HT'
        new_H (bool):
            If true, use new values for H-dev, shot > 12957

    Returns:
        ABCD (ndarray, float):
            The ABCD matrix

    Raises:
        NameError:
            If dev is not one of 'L', 'H', 'G', 'GT', 'GR', 'HT'
    """
    if dev not in ['L', 'H', 'G', 'GT', 'GR', 'HT']:
        raise NameError(f"Device is {dev:s}, but needs to be 'L', 'H', 'G', 'GT', 'GR', or 'HT'.")

    # ABCD matrix
    abcd = None
    if dev == 'L':
        sp = 3350 - Rinit * 1000  # [m] -> [mm]
        abcd = np.array([[1, 250 + sp], [0, 1]]).dot(
            np.array([[1, 0], [(1.52 - 1) / (-730), 1.52]])).dot(
            np.array([[1, 135], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (2700 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 1265 - LensZoom], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / 1100, 1.52]])).dot(
            np.array([[1, 40], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (-1100 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, LensZoom], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.52]])).dot(
            np.array([[1, 65], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (800 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 710 - LensFocus + 140], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / (-1270), 1.52]])).dot(
            np.array([[1, 90], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (1270 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 539 + 35 + LensFocus], [0, 1]]))

    elif dev == 'H':
        sp = 3350 - Rinit * 1000
        abcd = np.array([[1, 250 + sp], [0, 1]]).dot(
            np.array([[1, 0], [(1.52 - 1) / (- 730), 1.52]])).dot(
            np.array([[1, 135], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (2700 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 1265 - LensZoom], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / 1100, 1.52]])).dot(
            np.array([[1, 40], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (-1100 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, LensZoom], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.52]])).dot(
            np.array([[1, 65], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (800 * 1.52), 1 / 1.52]]))

        if new_H:
            abcd = abcd.dot(
                np.array([[1, 520 - LensFocus + 590 - 9.2], [0, 1]])).dot(
                np.array([[1, 0], [(1.52 - 1) / (-1100), 1.52]])).dot(
                np.array([[1, 88.4], [0, 1]])).dot(
                np.array([[1, 0], [(1 - 1.52) / (1100 * 1.52), 1 / 1.52]])).dot(
                np.array([[1, 446 + 35 + LensFocus - 9.2], [0, 1]]))
        else:
            abcd = abcd.dot(
                np.array([[1, 520 - LensFocus + 590], [0, 1]])).dot(
                np.array([[1, 0], [(1.52 - 1) / (-1400), 1.52]])).dot(
                np.array([[1, 70], [0, 1]])).dot(
                np.array([[1, 0], [(1 - 1.52) / (1400 * 1.52), 1 / 1.52]])).dot(
                np.array([[1, 446 + 35 + LensFocus], [0, 1]]))

    elif dev == 'G':
        sp = 3150 - Rinit * 1000
        abcd = np.array([[1, 1350 - LensZoom + sp], [0, 1]]).dot(
            np.array([[1, 0], [0, 1.545]])).dot(
            np.array([[1, 100], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.545) / (900 * 1.545), 1 / 1.545]])).dot(
            np.array([[1, 1430 - LensFocus + 660 + LensZoom + 470], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.545]])).dot(
            np.array([[1, 70], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.545) / (800 * 1.545), 1 / 1.545]])).dot(
            np.array([[1, LensFocus - 470], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.545]])).dot(
            np.array([[1, 80], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.545) / (800 * 1.545), 1 / 1.545]])).dot(
            np.array([[1, 390], [0, 1]]))

    elif dev == 'GT':
        sp = 2300 - Rinit * 1000
        abcd = np.array([[1, sp + (1954 - LensZoom)], [0, 1]]).dot(
            np.array([[1, 0], [(1.52 - 1) / (- 1000), 1.52]])).dot(
            np.array([[1, 160], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (1000 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 2280 - (1954 + 160 - LensZoom)], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / 1000, 1.52]])).dot(
            np.array([[1, 20], [0, 1]])).dot(
            np.array([[1, 0], [0, 1 / 1.52]])).dot(
            np.array([[1, 4288 - (2280 + 20) - LensFocus], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / (-1200), 1.52]])).dot(
            np.array([[1, 140], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (1200 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 4520 - (4288 + 140 - LensFocus)], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.52]])).dot(
            np.array([[1, 30], [0, 1]])).dot(
            np.array([[1, 0], [0, 1 / 1.52]])).dot(
            np.array([[1, 4940 - (4520 + 30)], [0, 1]]))

    elif dev == 'GR':
        sp = 2300 - Rinit * 1000
        abcd = np.array([[1, sp + (1954 - LensZoom)], [0, 1]]).dot(
            np.array([[1, 0], [(1.52 - 1) / (-1000), 1.52]])).dot(
            np.array([[1, 160], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (1000 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 2280 - (1954 + 160 - LensZoom)], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / 1000, 1.52]])).dot(
            np.array([[1, 20], [0, 1]])).dot(
            np.array([[1, 0], [0, 1 / 1.52]])).dot(
            np.array([[1, 4288 - (2280 + 20) - LensFocus], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / (- 1200), 1.52]])).dot(
            np.array([[1, 140], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (1200 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 4520 - (4288 + 140 - LensFocus)], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.52]])).dot(
            np.array([[1, 30], [0, 1]])).dot(
            np.array([[1, 0], [0, 1 / 1.52]])).dot(
            np.array([[1, 4940 - (4520 + 30)], [0, 1]]))

    elif dev == 'HT':
        sp = 2300 - Rinit * 1000
        abcd = np.array([[1, sp + 2586], [0, 1]]).dot(
            np.array([[1, 0], [0, 1.52]])).dot(
            np.array([[1, 140], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (770 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 4929 - (2586 + 140) - LensZoom], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / (1200), 1.52]])).dot(
            np.array([[1, 20], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (- 1200 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 5919 - (4929 + 20 - LensZoom) - LensFocus], [0, 1]])).dot(
            np.array([[1, 0], [(1.52 - 1) / (-1300), 1.52]])).dot(
            np.array([[1, 130], [0, 1]])).dot(
            np.array([[1, 0], [(1 - 1.52) / (1300 * 1.52), 1 / 1.52]])).dot(
            np.array([[1, 6489 - (5919 + 130 - LensFocus)], [0, 1]])).dot(
            np.array([[1, 0], [0, 1.52]])).dot(
            np.array([[1, 25.62], [0, 1]])).dot(
            np.array([[1, 0], [0, 1 / 1.52]])).dot(
            np.array([[1, 7094.62 - (6489 + 25.62)], [0, 1]]))

    return abcd


def get_geometry(cfg_diagnostic):
    """Builds channel geometry arrays.

    To re-construct the view, the following keys are extracted from the passed dictionary:
    * TFcurrent - Toroidal Field Coil current
    * LoFreq - ???
    * LensFocus - ???
    * LensZoom - ???
    * Mode - Either 'O' or 'X', ordinary/extra-ordinary
    * dev - In ['L', 'H', 'G', 'GT', 'GR', 'HT']

    Args:
        cfg_diagnostic (dict):
            Parameters section of diagnostic configuration.

    Returns:
        rarr (ndarray):
            Array containing radial coordinate of channels in m.
        zarr (ndarray):
            Array containing vertical coordinate of channels in m.
    """
    me = 9.1e-31             # electron mass, in kg
    e = 1.602e-19            # charge, in C
    mu0 = 4. * np.pi * 1e-7  # permeability
    ttn = 56 * 16              # total TF coil turns

    # Unpack ecei_cfg
    # Instead of TFcurrent multiplying by 1e3, put this in the config file
    TFcurrent = cfg_diagnostic["TFcurrent"]
    LoFreq = cfg_diagnostic["LoFreq"]
    LensFocus = cfg_diagnostic["LensFocus"]
    LensZoom = cfg_diagnostic["LensZoom"]
    # Set hn, depending on mode. If mode is undefined, set X-mode as default.
    hn = 2
    try:
        if cfg_diagnostic["Mode"] == 'O':
            hn = 1
        elif cfg_diagnostic["Mode"] == 'X':
            hn = 2

    except KeyError as k:
        print("ecei_cfg: key {0:s} not found. Defaulting to 2nd X-mode".format(k.__str__()))
        cfg_diagnostic["Mode"] = 'X'
        hn = 2

    # To vectorize calculation of the channel positions we flatten out
    # horizontal and vertical channel indices in horizontal order.
    arr_ch_hv = np.zeros([24 * 8, 2], dtype=int)
    for idx_v in range(1, 25):
        for idx_h in range(1, 9):
            ch = channel_2d(idx_v, idx_h, 24, 8, "horizontal")
            arr_ch_hv[ch.get_idx(), 0] = ch.ch_h
            arr_ch_hv[ch.get_idx(), 1] = ch.ch_v

    rpos_arr = hn * e * mu0 * ttn * TFcurrent /\
        (4. * np.pi * np.pi * me * ((arr_ch_hv[:, 0] - 1) * 0.9 + 2.6 + LoFreq) * 1e9)

    # With radial positions at hand, continue the calculations from beam_path
    # This is an (192, 2, 2) array, where the first dimension indices each individual channel
    abcd_array = np.array([get_abcd(LensFocus, LensZoom, rpos,
                                    cfg_diagnostic["dev"]) for rpos in rpos_arr])
    # vertical position from the reference axis (vertical center of all lens, z=0 line)
    zz = (np.arange(24, 0, -1) - 12.5) * 14  # [mm]
    # angle against the reference axis at ECEI array box
    aa = np.zeros_like(zz)

    # vertical posistion and angle at rpos
    za_array = np.dot(abcd_array, [zz, aa])

    zpos_arr = np.array([za_array[i, 0, v - 1] for
                         i, v in zip(np.arange(192), arr_ch_hv[:, 1])]) * 1e-3
    apos_arr = np.array([za_array[i, 1, v - 1] for i, v in zip(np.arange(192), arr_ch_hv[:, 1])])

    return(rpos_arr, zpos_arr, apos_arr)
# End of file kstar_ecei.py
