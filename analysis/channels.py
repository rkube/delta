# Coding: UTF-8 -*-

import itertools
import json

"""
Author: Ralph Kube

Defines abstraction for handling ECEI channels.

The ECEI diagnostics provides 192 independent channels that are arranged
into a 8 radial by 24 vertical view.

In fluctana, channels are referred to f.ex. L0101 or GT1507. The letters refer to a
device (D), the first two digits to the vertical channel number (V) and the last two digits
refer to the horizontal channel number (H). In delta we ue the format DDVVHH, where the letters
refer to one of the three components.


"""

def ch_num_to_vh(ch_num):
    """Returns a tuple (ch_v, ch_h) for a channel number. 
    Note that these are 1-based numbers.
    
    Parameters
    ----------
    ch_num: int, channel nu,ber.
    
    Returns:
    --------
    (ch_v, ch_h): int, Vertical and horizontal channel numbers.
    
    Vertical channel number is between 1 and 24. Horizontal channel number is 
    between 1 and 8.
    
    >>> ch_num_to_vh(17)
    (3, 1)
    """
    assert((ch_num >= 1) & (ch_num < 193))
    # Calculate using zero-base
    ch_num -= 1
    ch_v = ch_num // 8
    ch_h = ch_num - ch_v * 8
    return(ch_v + 1, ch_h + 1)


def ch_vh_to_num(ch_v, ch_h, debug=True):
    """Returns the linear channel index 1..192 for a ch_v, ch_h.
    
    Parameters:
    -----------
    ch_v, ch_h: int, vertical and horizontal chanel numbers
    debug: bool, if True, include assert tests for ch_h and ch_h.
    
    Returns:
    --------
    ch_num: int, linear channel index
    
    >>> ch_vh_to_num(12, 4)
    100
    """

    # We usually want to check that we are within the bounds.
    # But sometimes it is helpful to avoid this.
    if debug:
        assert((ch_v > 0) & (ch_v < 25))
        assert((ch_h > 0) & (ch_h < 9))

    return((ch_v - 1) * 8 + ch_h)


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.
    Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes"""

    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element



class channel():
    """Represents an ECEI channel.
    The ECEI array has 24 horizontal channels and 8 vertical channels.

    They are commonly represented as
    L2203
    where L denotes ???, 22 is the horizontal channel and 08 is the vertical channel.
    """

    def __init__(self, dev, ch_v, ch_h):
        """
        Parameters
        ----------
        dev: string, must be in 'L' 'H' 'G' 'GT' 'GR' 'HR'
        ch_h: int, Horizontal channel number, between 1 and 24
        ch_v: int, Vertical channel number, between 1 and 8
        """

        assert(dev in ['L', 'H', 'G', "GT", 'HT', 'GR', 'HR'])
        # 24 horizontal channels
        assert((ch_v > 0) & (ch_v  < 25))
        # 8 vertical channels
        assert((ch_h > 0) & (ch_h < 9))

        self.ch_v = ch_v
        self.ch_h = ch_h
        self.dev = dev
       
    
    @property
    def ch_num(self):
        return ch_vh_to_num(self.ch_v, self.ch_h)


    @classmethod
    def from_str(cls, ch_str):
        """Generates a channel object from a string, such as L2204 or GT1606.
        Input:
        ======
        cls: The class object (this is never passed to the method, but akin to self)
        ch_str: A channel string, such as L2205 of GT0808
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
        return(channel1)



    def __str__(self):
        """Prints the channel as a standardized string DDHHVV, where D is dev, H is ch_h and V is ch_v.
        DD can be 1 or 2 characters, H and V are zero-padded"""
        ch_str = "{0:s}{1:02d}{2:02d}".format(self.dev, self.ch_v, self.ch_h)

        return(ch_str)


    def __eq__(self, other):
        """Define equality for two channels when all three, dev, ch_h, and ch_v are equal to one another."""
        return (self.dev, self.ch_v, self.ch_h) == (other.dev, other.ch_v, other.ch_h)

    def idx(self):
        """Returns the linear, ZERO-BASED, index corresponding to ch_h and ch_v"""
        return ch_vh_to_num(self.ch_v, self.ch_h) - 1

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



class channel_pair:
    """Custom defined channel pair. 
    This is just a tuple with an overloaded equality operator. It's also hashable
    so one can use it in conjunction with sets

    >>> ch1 = channel('L', 13, 7)
    >>> ch2 = channel('L', 12, 7)
    >>> ch_pair = channel_pair(ch1, c2)

    >>> channel_pair(ch1, ch2) == channel_pair(ch2, c1)
    True

    The hash is that of a tuple consisting of the ordered channel indices of ch1 and ch2.
    """
    
    def __init__(self, ch1, ch2):
        self.ch1 = ch1
        self.ch2 = ch2


    def __eq__(self, other):
        return( ((self.ch1 == other.ch1) and (self.ch2 == other.ch2)) or
            ((self.ch1 == other.ch2) and (self.ch2 == other.ch1)))


    def __str__(self):
        """Returns a standardized string"""

        ch_str = f"{self.__class__.__name__}: (ch1={self.ch1}, ch2={self.ch2})"
        return(ch_str)


    def __iter__(self):
        yield from [self.ch1, self.ch2]

    def __hash__(self):
        """Implement hash so that we can use sets."""
        #print("({0:s} x {1:s}) hash={2:d}".format(self.ch1.__str__(), self.ch2.__str__(),\
        #    hash((min(self.ch1.idx(), self.ch2.idx()), max(self.ch1.idx(), self.ch2.idx()))))) 
        
        return hash((min(self.ch1.idx(), self.ch2.idx()), max(self.ch1.idx(), self.ch2.idx())))

    def to_json(self):
        return('{"ch1": ' + self.ch1.to_json() + ', "ch2": ' + self.ch2.to_json() + '}')

    
    @classmethod
    def from_json(cls, str):
        j = json.loads(str)
        ch1 = channel.from_json(json.dumps(j["ch1"]))
        ch2 = channel.from_json(json.dumps(j["ch2"]))
        cpair = cls(ch1, ch2)
        return cpair


class channel_range:
    """Defines iteration over classes. The iteration can be either linear or rectangular:

    For linear iteration, we map ch_h and ch_v to a linear index:
    index = (8 * (ch_v - 1)) + ch_h - 1, see ch_hv_to_num. 
    The iteration proceeds linear from the index of a start_channel to the index of an end_channel:
    >>> ch_start = channel('L', 1, 7)
    >>> ch_end = channel('L', 2,)

    """


    def __init__(self, ch_start, ch_end, mode="rectangle"):
        """Defines iteration over channels.
        Input:
        ======
        ch_start: channel, Start channel
        ch_end: channel, End channel
        mode: string, either 'linear' or 'rectangle'.

        If 'linear', the iterator moves linearly through the channel numbers from ch_start to ch_end.
        If 'rectangle', the iterator moves from ch_start.h to ch_end.h and from ch_start.v to ch_end.v

        yields channel object at the current position.
        """

        assert(ch_start.dev == ch_end.dev)

        self.ch_start = ch_start
        self.ch_end = ch_end

        self.dev = ch_start.dev
        self.ch_vi, self.ch_hi = ch_start.ch_v, ch_start.ch_h
        self.ch_vf, self.ch_hf = ch_end.ch_v, ch_end.ch_h

        self.ch_v = self.ch_vi
        self.ch_h = self.ch_hi
        
        assert(mode in ["linear", "rectangle"])
        self.mode = mode


    def __iter__(self):
        self.ch_v = self.ch_vi
        self.ch_h = self.ch_hi

        return(self)

    def __next__(self):

        # Test if we are on the last iteration
        if self.mode == "linear":
            # Remember to set debug=False so that we don't raise out-of-bounds errors
            # when generating the linear indices
            if(ch_vh_to_num(self.ch_v, self.ch_h, debug=False) > ch_vh_to_num(self.ch_vf, self.ch_hf, debug=False)):
                raise StopIteration

        elif self.mode == "rectangle":
            if((self.ch_v > self.ch_vf) | (self.ch_h > self.ch_hf)):
                raise StopIteration

        # IF we are not out of bounds, make a copy of the current v and h
        ch_v = self.ch_v
        ch_h = self.ch_h

        # Increase the horizontal channel and reset vertical channel number.
        # When iterating linearly, set v to 1
        if self.mode == "linear":
            if(self.ch_h == 8):
                self.ch_h = 1
                self.ch_v += 1
            else:
                self.ch_h += 1

        elif self.mode == "rectangle":
            if(self.ch_h == self.ch_hf):
                self.ch_h = self.ch_hi 
                self.ch_v += 1
            else:
                self.ch_h += 1

        # Return a channel with the previous ch_h 
        return channel(self.dev, ch_v, ch_h)


    @classmethod
    def from_str(cls, range_str, mode="rectangle"):
        """
        Generates a channel_range from a range, specified as
        'ECEI_[LGHT..][0-9]{4}-[0-9]{4}'
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
        ch_i = channel(dev, ch_vi, ch_hi)

        ch_f = int(m[1])
        ch_hf = ch_f % 100
        ch_vf = int(ch_f // 100)
        ch_f = channel(dev, ch_vf, ch_hf)

        return channel_range(ch_i, ch_f, mode)

    def length(self):
        """Calculates the number of channels in the list."""

        chnum_f = ch_vh_to_num(self.ch_vf, self.ch_hf)
        chnum_i = ch_vh_to_num(self.ch_vi, self.ch_hi)

        return(chnum_f - chnum_i + 1)


    def __str__(self):
        """Returns a standardized string"""

        ch_str = f"{self.__class__.__name__}: start: {self.ch_start}, end: {self.ch_end}, mode: {self.mode}"
        return(ch_str)

    
    def to_str(self):
        """Formats the channel list as f.ex. GT1207-2201"""

        #ch_str = "{0:s}{1:02d}{2:02d}-{3:02d}{4:02d}".format(self.dev, self.ch_hi, self.ch_vi, self.ch_hf, self.ch_vf)
        ch_str = f"{self.dev:s}{self.ch_vi:02d}{self.ch_hi:02d}-{self.ch_vf:02d}{self.ch_hf:02d}"
        return(ch_str)

    def to_json(self):
        return('{"ch_start": ' + ch_start.to_json() + ', "ch_end": ' + ch_end.to_json() + '}')



# End of file channels.py
