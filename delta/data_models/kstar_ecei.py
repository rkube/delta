# -*- Encoding: UTF-8 -*-

"""Contains helper function for working with the ECEI diagnostic.

Author: Minjun Choi (original), Ralph Kube (refactored)

These are just the member functions from kstarecei.py, copied here
so that we don't need to instantiate an kstarecei object.

Defines abstraction for handling ECEI channels.

The ECEI diagnostics provides 192 independent channels that are arranged
into a 8 radial by 24 vertical view.

In fluctana, channels are referred to f.ex. L0101 or GT1507. The letters refer to a
device (D), the first two digits to the vertical channel number (V) and the last two digits
refer to the horizontal channel number (H). In delta we ue the format DDVVHH, where the letters
refer to one of the three components.
"""


import numpy as np
import json

from data_models.channels_2d import channel_2d, channel_range


def channel_range_from_str(range_str):
    """Generates a channel_range from a range.

    Channel ranges are specified like this:

    ..code-block::
        'ECEI_[LGHT..][0-9]{4}-[0-9]{4}'

    Args:
        range_str (str):
        Specifices KSTAR ECEI channel range

    Returns:
        channel_range (channel_range):
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


class ecei_chunk():
    """Class that represents a time-chunk of ECEI data."""

    def __init__(self, data, tb, num_v=24, num_h=8):
        """Creates an ecei_chunk from a give dataset.

        Args:
            data (ndarray, float):
                Raw data for the ECEI voltages
            tb (timebase_streaming):
                timebase for ECEI voltages
            num_v (int):
                Number of vertical channels. Defaults to 24.
            num_h (int):
                Number of horizontal channels. Defaults to 8.

        Returns:
            None
        """
        # Data should have more than 1 dimension, last dimension is time
        assert(data.ndim > 1)
        #
        self.num_v, self.num_h = num_v, num_h
        # Data can be 2 or 3 dimensional
        assert(np.prod(data.shape[:-1]) == self.num_h * self.num_v)

        # We should ensure that the data is contiguous so that we can remove this from
        # if not data.flags.contiguous:
        # self.ecei_data = np.array(data, copy=True)
        self.ecei_data = np.require(data, dtype=np.float64, requirements=['C', 'O', 'W', 'A'])
        assert(self.ecei_data.flags.contiguous)

        self.tb = tb
        # Axis that indexes channels
        self.axis_ch = 0
        # Axis that indexes time
        self.axis_t = 1

    def data(self):
        """Common interface to data."""
        return self.ecei_data

    @property
    def shape(self):
        """Forwards to self.ecei_data.shape."""
        return self.ecei_data.shape

    def create_ft(self, fft_data, fft_params):
        """Returns a fourier-transformed object.

        Args:
            fft_data (ndarray):
                Numerical data
            fft_params (dict):
                Data passed to STFT function

        Returns:
            ecei_chunk_ft (ecei_chunk_ft):
                Chunk of Fourier-transformed data
        """
        return ecei_chunk_ft(fft_data, tb=self.tb,
                             freqs=None, fft_params=fft_params)


class ecei_chunk_ft():
    """Represents a fourier-transformed time-chunk of ECEI data."""

    def __init__(self, data_ft, tb, freqs, fft_params, axis_ch=0, axis_t=1, num_v=24, num_h=8):
        """Initializes with data and meta-information.

        Args:
            data_ft (ndarray, float):
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
        self.data_ft = data_ft
        self.tb = tb
        self.freqs = freqs
        self.fft_params = fft_params
        self.axis_ch = axis_ch
        self.axis_t = axis_t
        self.num_v = num_v
        self.num_h = num_h

    def data(self):
        """Common interface to data."""
        return self.data_ft

    @property
    def shape(self):
        """Forwards to self.ecei_data.shape."""
        return self.ecei_data.shape


class ecei_channel_2d(channel_2d):
    """Represents an ECEI channel.

    The ECEI array has 24 horizontal channels and 8 vertical channels.

    They are commonly represented as
    L2203
    where L denotes ???, 22 is the horizontal channel and 08 is the vertical channel.
    """

    def __init__(self, dev, ch_v, ch_h):
        """Initializes the channel.

        Args:
            dev (string):
                must be in 'L' 'H' 'G' 'GT' 'GR' 'HR'
            ch_h (int):
                Horizontal channel number, between 1 and 24
            ch_v (int):
                Vertical channel number, between 1 and 8

        Returns:
            None
        """
        #
        assert(dev in ['L', 'H', 'G', "GT", 'HT', 'GR', 'HR'])
        self.dev = dev
        super().__init__(ch_v, ch_h, 24, 8)

    @classmethod
    def from_str(cls, ch_str):
        """Generates a channel object from a string, such as L2204 or GT1606.

        Args:
            cls:
                The class object (this is never passed to the method, but akin to self)
            ch_str (string):
                A channel string, such as L2205 of GT0808

        Returns:
            channel1 (ecei_channel_2d):
                Newly instantiated ecei_channel_2d object
        """
        import re
        # Define a regular expression that matches a sequence of 1 to 2 characters in [A-Z]
        # This will be our dev.
        m = re.search('[A-Z]{1,2}', ch_str)
        try:
            dev = m.group(0)
        except NameError:
            raise AttributeError("Could not parse channel string " + ch_str)

        # Define a regular expression that matches 4 consecutive digits
        # These will be used to calculate ch_h and ch_v
        m = re.search('[0-9]{4}', ch_str)
        ch_num = int(m.group(0))

        ch_h = (ch_num % 100)
        ch_v = int(ch_num // 100)

        channel1 = cls(dev, ch_v, ch_h)
        return channel1

    def __str__(self):
        """Prints the channel as a standardized string DDHHVV.

        Here D is dev, H is ch_h and V is ch_v.
        DD can be 1 or 2 characters, H and V are zero-padded.
        """
        ch_str = "{0:s}{1:02d}{2:02d}".format(self.dev, self.ch_v, self.ch_h)

        return(ch_str)

    def to_json(self):
        """Returns the class in JSON notation.

        This method avoids serialization error when using non-standard int types,
        such as np.int64 etc...
        """
        d = {"ch_v": int(self.ch_v), "ch_h": int(self.ch_h),
             "dev": self.dev, "ch_num": int(self.ch_num)}
        return json.dumps(d, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, str):
        """Returns a channel instance from a json string."""
        j = json.loads(str)

        channel1 = cls(j["dev"], int(j["ch_v"]), int(j["ch_h"]))
        return(channel1)


def get_abcd(channel, LensFocus, LensZoom, Rinit, new_H=True):
    """Returns ABCD matrix for KSTAR ECEI diagnostic.

    Args:
        ch (channel):
            channel
        LensZoom (float):
            LensZoom
        LensFocus (float):
            LensFocus
        Rinit (float):
            Radial position of the channel, in meter
        new_H (bool):
            If true, use new values for H-dev, shot > 12957

    Returns:
        ABCD (ndarray, float):
            The ABCD matrix

    Raises:
        NameError:
            If channel is not one of 'L', 'H', 'G', 'GT', 'GR', 'HT'
    """
    # ABCD matrix
    abcd = None
    if channel.dev == 'L':
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

    elif channel.dev == 'H':
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

    elif channel.dev == 'G':
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

    elif channel.dev == 'GT':
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

    elif channel.dev == 'GR':
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

    elif channel.dev == 'HT':
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
    else:
        raise ValueError("Channel needs to be either 'L', 'H', 'G', 'GT', 'GR', or 'HT'. ")

    return abcd


def beam_path(ch, LensFocus, LensZoom, rpos):
    """Calculates the ray vertical position and angle at rpos.

    Starting from the array box position.

    Args:
        ch (channel):
            Channel for which to calculate the beam path
        LensFocus (float):
            LensFocus factor
        LensZoom (float):
            LensZoom factor.
        rpos (float):
            Radial position of the channel view, in meters
        ch_v (int):
            number of vertical channel

    Returns:
        zpos (float):
            Z-coordinate of beam-path
        apos (float):
            angle coordinate of beam-path
    """
    abcd = get_abcd(ch, LensFocus, LensZoom, rpos)

    # vertical position from the reference axis (vertical center of all lens, z=0 line)
    zz = (np.arange(24, 0, -1) - 12.5) * 14  # [mm]
    # angle against the reference axis at ECEI array box
    aa = np.zeros(np.size(zz))

    # vertical posistion and angle at rpos
    za = np.dot(abcd, [zz, aa])
    zpos = za[0][ch.ch_v - 1] * 1e-3  # zpos [m]
    # angle [rad] positive means the (z+) up-directed (divering from array to plasma)
    apos = za[1][ch.ch_v - 1]

    return zpos, apos


def channel_position(ch, ecei_cfg):
    """Calculates the position of a channel in configuration space.

    Args:
        ch (channel):
            The channel whos position we want to calculate
        ecei_cfg (dict):
            Parameters of the ECEi diagnostic.

    Returns:
        rpos (float):
            R-position of ECEI channel, in m.
        zpos (float):
            Z-position of ECEI channel, in m.
        apos (float):
            angle of ECEI channel. In radians, I think?
    """
    me = 9.1e-31             # electron mass, in kg
    e = 1.602e-19            # charge, in C
    mu0 = 4. * np.pi * 1e-7  # permeability
    ttn = 56 * 16              # total TF coil turns

    # Unpack ecei_cfg
    # Instead of TFcurrent multiplying by 1e3, put this in the config file
    TFcurrent = ecei_cfg["TFcurrent"]
    LoFreq = ecei_cfg["LoFreq"]
    LensFocus = ecei_cfg["LensFocus"]
    LensZoom = ecei_cfg["LensZoom"]
    # Set hn, depending on mode. If mode is undefined, set X-mode as default.
    try:
        if ecei_cfg["Mode"] == 'O':
            hn = 1
        elif ecei_cfg["Mode"] == 'X':
            hn = 2

    except KeyError as k:
        print("ecei_cfg: key {0:s} not found. Defaulting to 2nd X-mode".format(k.__str__()))
        ecei_cfg["Mode"] = 'X'
        hn = 2

    rpos = hn * e * mu0 * ttn * TFcurrent /\
        (4. * np.pi * np.pi * me * ((ch.ch_h - 1) * 0.9 + 2.6 + LoFreq) * 1e9)
    zpos, apos = beam_path(ch, LensFocus, LensZoom, rpos)
    return (rpos, zpos, apos)


# End of file kstar_ecei.py
