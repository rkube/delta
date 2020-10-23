# -*- Encoding: UTF-8 -*-

"""
Author: Minjun Choi (original), Ralph Kube (refactored)

Contains helper function for working with the ECEI diagnostic.
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


#from data_models.channels_2d import channel_2d, channel_range
from .channels_2d import channel_2d, channel_range



def channel_range_from_str(range_str):
        """
        Generates a channel_range from a range, specified as

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
        try:
            dev = m.group(0)
        except:
            raise AttributeError("Could not parse channel string {0:s}".format(range_str))

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
    """Class that represents a time-chunk of ECEI data"""

    def __init__(self, data, tb, num_v=24, num_h=8):
        """
        Creates an ecei_chunk from a give dataset

        Args:
            data (ndarray, float): 
                Raw data for the ECEI voltages
            tb (timebase_streaming): 
                timebase for ECEI voltages
            num_v (int): 
                Number of vertical channels. Defaults to 24.
            num_h (int): 
                Number of horizontal channels. Defaults to 8.
        """

        # Data should have more than 1 dimension, last dimension is time
        assert(data.ndim > 1)
        #
        self.num_v, self.num_h = num_v, num_h
        # Data can be 2 or 3 dimensional
        assert(np.prod(data.shape[:-1]) == self.num_h * self.num_v)

        # We should ensure that the data is contiguous so that we can remove this from
        #if not data.flags.contiguous:
        #self.ecei_data = np.array(data, copy=True)
        self.ecei_data = np.require(data, dtype=np.float64, requirements=['C', 'O', 'W', 'A'])
        assert(self.ecei_data.flags.contiguous)

        self.tb = tb
        # Axis that indexes channels
        self.axis_ch = 0
        # Axis that indexes time
        self.axis_t = 1

    def data(self):
        """Common interface to data"""
        return self.ecei_data


    def create_ft(self, fft_data, fft_params):
        """Returns a fourier-transformed object"""
        return ecei_chunk_ft(fft_data, tb=self.tb, 
                             freqs=None, fft_params=fft_params)


class ecei_chunk_ft():
    """Class that represents a fourier-transformed time-chunk of ECEI data"""

    def __init__(self, data_ft, tb, freqs, fft_params, axis_ch=0, axis_t=1, num_v=24, num_h=8):
        """Creates a fourier-transformed time-chunk of ECEI data

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
        """Common interface to data"""
        return self.data_ft


class ecei_channel_2d(channel_2d):
    """Represents an ECEI channel.
    The ECEI array has 24 horizontal channels and 8 vertical channels.

    They are commonly represented as
    L2203
    where L denotes ???, 22 is the horizontal channel and 08 is the vertical channel.
    """

    def __init__(self, dev, ch_v, ch_h):
        """
        Args: 
            dev (string): 
                must be in 'L' 'H' 'G' 'GT' 'GR' 'HR'
            ch_h (int): 
                Horizontal channel number, between 1 and 24
            ch_v (int): 
                Vertical channel number, between 1 and 8
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
        """

        import re
        # Define a regular expression that matches a sequence of 1 to 2 characters in [A-Z]
        # This will be our dev.
        m = re.search('[A-Z]{1,2}', ch_str)
        try:
            dev = m.group(0)
        except:
            raise AttributeError("Could not parse channel string {0:s}".format(ch_str))

        # Define a regular expression that matches 4 consecutive digits
        # These will be used to calculate ch_h and ch_v
        m = re.search('[0-9]{4}', ch_str)
        ch_num = int(m.group(0))

        ch_h = (ch_num % 100)
        ch_v = int(ch_num // 100)

        channel1 = cls(dev, ch_v, ch_h)
        return channel1


    def __str__(self):
        """Prints the channel as a standardized string DDHHVV, where D is dev, H is ch_h and V is ch_v.
        DD can be 1 or 2 characters, H and V are zero-padded"""
        ch_str = "{0:s}{1:02d}{2:02d}".format(self.dev, self.ch_v, self.ch_h)

        return(ch_str)


    def to_json(self):
        """Returns the class in JSON notation.
           This method avoids serialization error when using non-standard int types,
           such as np.int64 etc..."""
        d = {"ch_v": int(self.ch_v), "ch_h": int(self.ch_h), "dev": self.dev, "ch_num": int(self.ch_num)}
        return json.dumps(d, default=lambda o: o.__dict__, sort_keys=True, indent=4)


    @classmethod
    def from_json(cls, str):
        """Returns a channel instance from a json string"""

        j = json.loads(str)

        channel1 = cls(j["dev"], int(j["ch_v"]), int(j["ch_h"]))
        return(channel1)



def get_abcd(channel, LensFocus, LensZoom, Rinit, new_H=True):
    """

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
    """

    # ABCD matrix
    if channel.dev == 'L':
        sp = 3350 - Rinit*1000  # [m] -> [mm]
        abcd = np.array([[1,250+sp],[0,1]]).dot(
               np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
               np.array([[1,135],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
               np.array([[1,1265-LensZoom],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
               np.array([[1,40],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
               np.array([[1,LensZoom],[0,1]])).dot(
               np.array([[1,0],[0,1.52]])).dot(
               np.array([[1,65],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]])).dot(
               np.array([[1,710-LensFocus+140],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(-1270),1.52]])).dot(
               np.array([[1,90],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1270*1.52),1/1.52]])).dot(
               np.array([[1,539+35+LensFocus],[0,1]]))

    elif channel.dev == 'H':
        sp = 3350 - Rinit*1000
        abcd = np.array([[1,250+sp],[0,1]]).dot(
               np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
               np.array([[1,135],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
               np.array([[1,1265-LensZoom],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
               np.array([[1,40],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
               np.array([[1,LensZoom],[0,1]])).dot(
               np.array([[1,0],[0,1.52]])).dot(
               np.array([[1,65],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]]))

        if new_H:
            abcd = abcd.dot(
               np.array([[1,520-LensFocus+590-9.2],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(-1100),1.52]])).dot(
               np.array([[1,88.4],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1100*1.52),1/1.52]])).dot(
               np.array([[1,446+35+LensFocus-9.2],[0,1]]))
        else:
            abcd = abcd.dot(
               np.array([[1,520-LensFocus+590],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(-1400),1.52]])).dot(
               np.array([[1,70],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1400*1.52),1/1.52]])).dot(
               np.array([[1,446+35+LensFocus],[0,1]]))

    elif channel.dev == 'G':
        sp = 3150 - Rinit*1000
        abcd = np.array([[1,1350-LensZoom+sp],[0,1]]).dot(
               np.array([[1,0],[0,1.545]])).dot(
               np.array([[1,100],[0,1]])).dot(
               np.array([[1,0],[(1-1.545)/(900*1.545),1/1.545]])).dot(
               np.array([[1,1430-LensFocus+660+LensZoom+470],[0,1]])).dot(
               np.array([[1,0],[0,1.545]])).dot(
               np.array([[1,70],[0,1]])).dot(
               np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
               np.array([[1,LensFocus-470],[0,1]])).dot(
               np.array([[1,0],[0,1.545]])).dot(
               np.array([[1,80],[0,1]])).dot(
               np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
               np.array([[1,390],[0,1]]))

    elif channel.dev == 'GT':
        sp = 2300 - Rinit*1000
        abcd = np.array([[1,sp+(1954-LensZoom)],[0,1]]).dot(
               np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
               np.array([[1,160],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
               np.array([[1,2280-(1954+160-LensZoom)],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
               np.array([[1,20],[0,1]])).dot(
               np.array([[1,0],[0,1/1.52]])).dot(
               np.array([[1,4288-(2280+20)-LensFocus],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
               np.array([[1,140],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
               np.array([[1,4520-(4288+140-LensFocus)],[0,1]])).dot(
               np.array([[1,0],[0,1.52]])).dot(
               np.array([[1,30],[0,1]])).dot(
               np.array([[1,0],[0,1/1.52]])).dot(
               np.array([[1,4940-(4520+30)],[0,1]]))

    elif channel.dev == 'GR':
        sp = 2300 - Rinit*1000
        abcd = np.array([[1,sp+(1954-LensZoom)],[0,1]]).dot(
               np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
               np.array([[1,160],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
               np.array([[1,2280-(1954+160-LensZoom)],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
               np.array([[1,20],[0,1]])).dot(
               np.array([[1,0],[0,1/1.52]])).dot(
               np.array([[1,4288-(2280+20)-LensFocus],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
               np.array([[1,140],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
               np.array([[1,4520-(4288+140-LensFocus)],[0,1]])).dot(
               np.array([[1,0],[0,1.52]])).dot(
               np.array([[1,30],[0,1]])).dot(
               np.array([[1,0],[0,1/1.52]])).dot(
               np.array([[1,4940-(4520+30)],[0,1]]))

    elif channel.dev == 'HT':
        sp = 2300 - Rinit*1000
        abcd = np.array([[1,sp+2586],[0,1]]).dot(
               np.array([[1,0],[0,1.52]])).dot(
               np.array([[1,140],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(770*1.52),1/1.52]])).dot(
               np.array([[1,4929-(2586+140)-LensZoom],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(1200),1.52]])).dot(
               np.array([[1,20],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(-1200*1.52),1/1.52]])).dot(
               np.array([[1,5919-(4929+20-LensZoom)-LensFocus],[0,1]])).dot(
               np.array([[1,0],[(1.52-1)/(-1300),1.52]])).dot(
               np.array([[1,130],[0,1]])).dot(
               np.array([[1,0],[(1-1.52)/(1300*1.52),1/1.52]])).dot(
               np.array([[1,6489-(5919+130-LensFocus)],[0,1]])).dot(
               np.array([[1,0],[0,1.52]])).dot(
               np.array([[1,25.62],[0,1]])).dot(
               np.array([[1,0],[0,1/1.52]])).dot(
               np.array([[1,7094.62-(6489+25.62)],[0,1]]))

    return abcd


def beam_path(ch, LensFocus, LensZoom, rpos):
    """Calculates the ray vertical position and angle at rpos [m] ray starting from the array box position.

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
    """

    abcd = get_abcd(ch, LensFocus, LensZoom, rpos)

    # vertical position from the reference axis (vertical center of all lens, z=0 line) at ECEI array box
    zz = (np.arange(24, 0, -1) - 12.5) * 14  # [mm]
    # angle against the reference axis at ECEI array box
    aa = np.zeros(np.size(zz))

    # vertical posistion and angle at rpos
    za = np.dot(abcd, [zz, aa])
    zpos = za[0][ch.ch_v - 1] * 1e-3  # zpos [m]
    apos = za[1][ch.ch_v - 1]  # angle [rad] positive means the (z+) up-directed (divering from array to plasma)

    return zpos, apos

def channel_position(ch, ecei_cfg):
    """Calculates the position of a channel in configuration space

    Args:
        ch (channel): 
            The channel whos position we want to calculate
        ecei_cfg (dict): 
            Parameters of the ECEi diagnostic.
    """

    me = 9.1e-31             # electron mass, in kg
    e = 1.602e-19            # charge, in C
    mu0 = 4. * np.pi * 1e-7  # permeability
    ttn = 56*16              # total TF coil turns

    # Unpack ecei_cfg
    TFcurrent = ecei_cfg["TFcurrent"] # Instead of multiplying by 1e3, we put this in the config file
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




# class ecei_channel():
#     """Defines data as an ECEI channel"""
#
#     def __init__(self, data, tb, channel, t_offset=None, t_crop=None):
#         """
#         Parameters:
#         -----------
#         data: ndarray, float - Raw Voltages from the ECEI diagnostic
#         tb: timebase - Timebase object for the raw voltages
#         channel: channel object - Describes the channel
#         t_offset: tuple (t_n0, t_n1) - Tuple that defines the time interval where a signal reference value is calculated. If None,
#                                      raw values will be used.
#         t_crop: tuple (t_c0, t_c1) - Defines the time interval where the data is cropped to. If None, data will not
#                                      be cropped
#
#         """
#
#         # Make sure that the timebase is appropriate for the data
#         assert(np.max(data.shape) == tb.num_samples)
#         self.ecei_data = data * 1e-4
#         self.tb_raw = tb
#         self.channel = channel
#
#         self.is_cropped = False
#         self.is_normalized = False
#
#         if t_offset is not None:
#             # Calculate the signal offset
#             self.calculate_offsets(t_offset)
#
#             if t_crop is not None:
#                 self.crop_data(t_crop)
#                 self.is_cropped = True
#
#             # Subtract signal offset after signal has been cropped
#             self.ecei_data = (self.ecei_data - self.offlev)
#
#         else:
#             if t_crop is not None:
#                 self.crop_data(t_crop)
#                 self.is_cropped = True
#
#         self.siglev = np.median(self.ecei_data)
#         self.sigstd = self.ecei_data.std()
#
#         # After signal is shifted and cropped we normalize the signal
#         self.ecei_data = self.ecei_data / self.ecei_data.mean() - 1.0
#
#         print("all good")
#
#
#     def calculate_offsets(self, t_offset):
#         """Calculate mean and standard deviation from un-normalized channel data
#         Parameters:
#         -----------
#         t_norm: t_n0, t_n1) - Tuple that defines the time interval where the data is normalized to
#         """
#
#         if self.is_normalized == False:
#             # Calculate normalization constants. See fluctana.py, line 118ff
#             idx_norm = [self.tb_raw.time_to_idx(t) for t in t_offset]
#
#             offset_interval = self.ecei_data[idx_norm[0]:idx_norm[1]]
#             print(f"Calculating offsets at {idx_norm[0]:d}:{idx_norm[1]:d}")
#             self.offlev = np.median(offset_interval)
#             self.offstd = offset_interval.std()
#
#
#     def calculate_sigstats(self):
#         """Calculate signal statistics. Before normalization.
#         """
#
#         assert(self.is_normalized == False)
#         self.siglev = np.median(self.ecei_data)
#         self.sigstd = self.ecei_data.std()
#
#
#     def crop_data(self, t_crop):
#         """Crops the data to the interval defined by t_crop.
#
#         Input:
#         ======
#         t_crop: Tuple (t0, t1), where t0 and t1 correspond to the timebase passed into __init__
#         """
#         if self.is_cropped == False:
#             idx_crop = [self.tb_raw.time_to_idx(t) for t in t_crop]
#             print("Cropping data using ", idx_crop)
#             self.ecei_data = self.ecei_data[idx_crop[0]:idx_crop[1]]
#             print(f"data[0] = {self.ecei_data[0]}, data[-1] = {self.ecei_data[-1]}")
#
#
#     def position(self):
#         # Returns the R,Z position of the channel
#         pass
#
#     def channel_name(self):
#         # Returns the channel name
#         pass
#
#     def get_timebase(self):
#         """Generates and returns a timebase object for the channel data"""
#
#         pass
#
#     def check_signal_level(self):
#         """
#         Returns True if the signal level is acceptable
#         Returns False if the signal level is bad
#         """
#
#         # Signal level is defined as the median, see fluctana.py line
#
#         if self.siglev > 0.01:
#             ref = 100. * self.offstd / self.siglev
#         else:
#             ref = 100.
#
#         if ref > 30.0:
#             warnings.warn(f"LOW signal level channel {self.channel:s}, ref = {ref:4.1f}, siglevel = {self.siglev:4.1f} V")
#             return False
#
#         return True
#
#
#     def check_bottom_sat(self):
#         """
#         Check bottom saturation.
#         Good saturation: Return True if bottom saturation is above 0.001
#         Bad saturation: Return False if bottom saturation is below 0.001
#         """
#
#         if self.offstd < 0.0001:
#             warning.warn(f"SAT offstd data channel {self.channel:s}, offstd = {self.offstd:g}%, offlevel = {self.offlev:g} V")
#             return False
#
#         return True
#
#
#     def check_top_sat(self):
#         """
#         Check top saturation.
#         Good saturation: Return True if top saturation is above 0.001
#         Bad saturation: Return False if top saturation is below 0.001
#         """
#
#         if self.sigstd < 0.001:
#             warning.warn(f"SAT sigstd data channel {self.channel:s}, sigstd = {self.sigstd:g}%, offlevel = {self.offlev:g} V")
#             return False
#
#         return True
#
#
#     def data(self):
#         """Common interface to data"""
#         return self.ecei_data



# class channel_range:
#     """Defines iteration over classes. The iteration can be either linear or rectangular:
#
#     For linear iteration, we map ch_h and ch_v to a linear index:
#     index = (8 * (ch_v - 1)) + ch_h - 1, see ch_hv_to_num.
#     The iteration proceeds linear from the index of a start_channel to the index of an end_channel:
#     >>> ch_start = channel('L', 1, 7)
#     >>> ch_end = channel('L', 2,)
#
#     """
#
#
#     def __init__(self, ch_start, ch_end, mode="rectangle"):
#         """Defines iteration over channels.
#         Input:
#         ======
#         ch_start: channel, Start channel
#         ch_end: channel, End channel
#         mode: string, either 'linear' or 'rectangle'.
#
#         If 'linear', the iterator moves linearly through the channel numbers from ch_start to ch_end.
#         If 'rectangle', the iterator moves from ch_start.h to ch_end.h and from ch_start.v to ch_end.v
#
#         yields channel object at the current position.
#         """
#
#         assert(ch_start.dev == ch_end.dev)
#
#         self.ch_start = ch_start
#         self.ch_end = ch_end
#
#         self.dev = ch_start.dev
#         self.ch_vi, self.ch_hi = ch_start.ch_v, ch_start.ch_h
#         self.ch_vf, self.ch_hf = ch_end.ch_v, ch_end.ch_h
#
#         self.ch_v = self.ch_vi
#         self.ch_h = self.ch_hi
#
#         assert(mode in ["linear", "rectangle"])
#         self.mode = mode
#
#
#     def __iter__(self):
#         self.ch_v = self.ch_vi
#         self.ch_h = self.ch_hi
#
#         return(self)
#
#     def __next__(self):
#
#         # Test if we are on the last iteration
#         if self.mode == "linear":
#             # Remember to set debug=False so that we don't raise out-of-bounds errors
#             # when generating the linear indices
#             if(ch_vh_to_num(self.ch_v, self.ch_h, debug=False) > ch_vh_to_num(self.ch_vf, self.ch_hf, debug=False)):
#                 raise StopIteration
#
#         elif self.mode == "rectangle":
#             if((self.ch_v > self.ch_vf) | (self.ch_h > self.ch_hf)):
#                 raise StopIteration
#
#         # IF we are not out of bounds, make a copy of the current v and h
#         ch_v = self.ch_v
#         ch_h = self.ch_h
#
#         # Increase the horizontal channel and reset vertical channel number.
#         # When iterating linearly, set v to 1
#         if self.mode == "linear":
#             if(self.ch_h == 8):
#                 self.ch_h = 1
#                 self.ch_v += 1
#             else:
#                 self.ch_h += 1
#
#         elif self.mode == "rectangle":
#             if(self.ch_h == self.ch_hf):
#                 self.ch_h = self.ch_hi
#                 self.ch_v += 1
#             else:
#                 self.ch_h += 1
#
#         # Return a channel with the previous ch_h
#         return channel(self.dev, ch_v, ch_h)
#
#
#     @classmethod
#     def from_str(cls, range_str, mode="rectangle"):
#         """
#         Generates a channel_range from a range, specified as
#         'ECEI_[LGHT..][0-9]{4}-[0-9]{4}'
#         """
#
#         import re
#
#         m = re.search('[A-Z]{1,2}', range_str)
#         try:
#             dev = m.group(0)
#         except:
#             raise AttributeError("Could not parse channel string {0:s}".format(range_str))
#
#         m = re.findall('[0-9]{4}', range_str)
#
#         ch_i = int(m[0])
#         ch_hi = ch_i % 100
#         ch_vi = int(ch_i // 100)
#         ch_i = channel(dev, ch_vi, ch_hi)
#
#         ch_f = int(m[1])
#         ch_hf = ch_f % 100
#         ch_vf = int(ch_f // 100)
#         ch_f = channel(dev, ch_vf, ch_hf)
#
#         return channel_range(ch_i, ch_f, mode)
#
#     def length(self):
#         """Calculates the number of channels in the list."""
#
#         chnum_f = ch_vh_to_num(self.ch_vf, self.ch_hf)
#         chnum_i = ch_vh_to_num(self.ch_vi, self.ch_hi)
#
#         return(chnum_f - chnum_i + 1)
#
#
#     def __str__(self):
#         """Returns a standardized string"""
#
#         ch_str = f"{self.__class__.__name__}: start: {self.ch_start}, end: {self.ch_end}, mode: {self.mode}"
#         return(ch_str)
#
#
#     def to_str(self):
#         """Formats the channel list as f.ex. GT1207-2201"""
#
#         #ch_str = "{0:s}{1:02d}{2:02d}-{3:02d}{4:02d}".format(self.dev, self.ch_hi, self.ch_vi, self.ch_hf, self.ch_vf)
#         ch_str = f"{self.dev:s}{self.ch_vi:02d}{self.ch_hi:02d}-{self.ch_vf:02d}{self.ch_hf:02d}"
#         return(ch_str)
#
#     def to_json(self):
#         return('{"ch_start": ' + ch_start.to_json() + ', "ch_end": ' + ch_end.to_json() + '}')
#




# class ecei_view():
#     """Defines the view of an ECEI. This extends ecei_channel to the entire diagnostic

#     Parameters:
#     -----------
#     datafilename: string, filename to the HDF5 file
#     tb: timebase - Timebase object for the raw voltages
#     dev: device name
#     t_offset: tuple (t_n0, t_n1) - Tuple that defines the time interval where a signal reference value is calculated. If None,
#                                     raw values will be used.
#     t_crop: tuple (t_c0, t_c1) - Defines the time interval where the data is cropped to. If None, data will not
#                                     be cropped


#     """

#     def __init__(self, datafilename, tb, dev, t_offset=(-0.099, -0.089), t_crop=(1.0, 1.1), num_v=24, num_h=8):

#         # Number of vertical and horizontal channels
#         self.num_v = num_v
#         self.num_h = num_h
#         # Infer number of samples in the cropped interval
#         idx_offset = [tb.time_to_idx(t) for t in t_offset]
#         if idx_crop is not None:
#             idx_crop = [tb.time_to_idx(t) for t in t_crop]
#         self.num_samples = idx_crop[1] - idx_crop[0]

#         # Use float32 since data is generated from 16bit integers
#         self.ecei_data = np.zeros([self.num_v, self.num_h, self.num_samples], dtype=np.float32)
#         # Marks data the we ignore for plotting etc.
#         self.bad_data = np.zeros([self.num_v, self.num_h], dtype=bool)

#         # Offset level
#         self.offlev = np.zeros([self.num_v, self.num_h], dtype=np.float32)
#         # Offset standard deviation
#         self.offstd = np.zeros([self.num_v, self.num_h], dtype=np.float32)
#         # Signal level
#         self.siglev = np.zeros([self.num_v, self.num_h], dtype=np.float32)
#         # Signal standard deviation
#         self.sigstd = np.zeros([self.num_v, self.num_h], dtype=np.float32)

#         tic = time.perf_counter()
#         # Load data from HDF file
#         with h5py.File(datafilename, "r") as df:
#             print("Trigger time: ", df['ECEI'].attrs['TriggerTime'])
#             for ch_idx in range(192):
#                 ch_v, ch_h = np.mod(ch_idx, 24), ch_idx // 24
#                 ch_str = f"/ECEI/ECEI_{dev}{(ch_v + 1):02d}{(ch_h + 1):02d}/Voltage"

#                 # Calculate the start-of-shot offset
#                 self.offlev[ch_v, ch_h] = np.median(df[ch_str][idx_offset[0]:idx_offset[1]]) * 1e-4
#                 self.offstd[ch_v, ch_h] = df[ch_str][idx_offset[0]:idx_offset[1]].std() * 1e-4

#                 tmp = df[ch_str][idx_crop[0]:idx_crop[1]] * 1e-4  - self.offlev[ch_v, ch_h]

#                 self.siglev[ch_v, ch_h] = np.median(tmp)
#                 self.sigstd = tmp.std()
#                 self.ecei_data[ch_v, ch_h, :] = tmp / tmp.mean() - 1.0

#         toc = time.perf_counter()

#         print(f"Loading data took {(toc - tic):4.2f}s")

#         self.tb = timebase(t_crop[0], t_crop[1], tb.f_sample)

#         self.mark_bad_channels(verbose=True)


#     def mark_bad_channels(self, verbose=False):
#         """Mark bad channels. These are channels with either
#         * Low signal level: std(offset) / siglev > 0.3
#         * Saturated signal data(bottom saturation): std(offset) < 0.001
#         * Saturated offset data(top saturation): std(signal) < 0.001
#         """

#         # Check for low signal level
#         ref = 100. * self.offstd / self.siglev
#         ref[self.siglev < 0.01] = 100

#         if verbose:
#             for item in np.argwhere(ref > 30.0):
#                 print(f"LOW SIGNAL: channel({item[0] + 1:d},{item[1] + 1:d}): {ref[tuple(item)]*1e2:4.2f}")
#         self.bad_data[ref > 30.0] = True

#         # Mark bottom saturated channels
#         self.bad_data[self.offstd < 1e-3] = True
#         if verbose:
#             for item in np.argwhere(self.offstd < 1e-3):
#                 os = self.offstd[tuple(item)]
#                 ol = self.offlev[tuple(item)]
#                 print(f"SAT offset data channel ({item[0] + 1:d}, {item[1] + 1:d}) offstd = {os} offlevel = {ol}")

#         # Mark top saturated channels
#         self.bad_data[self.sigstd < 1e-3] = True
#         if verbose:
#             for item in np.argwhere(self.sigstd < 1e-3):
#                 os = self.offstd[tuple(item)]
#                 ol = self.offlev[tuple(item)]
#                 print(f"SAT signal data channel ({item[0] + 1:d}, {item[1] + 1:d}) offstd = {os} offlevel = {ol}")


# End of file kstar_ecei.py
