# Coding: UTF-8 -*-

def ch_num_to_hv(ch_num):
    """Returns a tuple (ch_h, ch_v) for a channel number"""
    assert((ch_num > 0) & (ch_num < 193))
    ch_h = ch_num // 8
    ch_v = ch_num - ch_h * 8
    return(ch_h, ch_v)


def ch_hv_to_num(ch_h, ch_v, debug=False):
    """Returns the channel number 1..192 from a channel h and v"""
    if debug:
        assert((ch_v > 0) & (ch_v < 9))
        assert((ch_h > 0) & (ch_h < 25))

    return(ch_h * 8 + ch_v)


class channel_list:
    def __init__(self, ch_start, ch_end, mode="linear"):
        """Generates an iterator over channels.
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
        self.dev = ch_start.dev
        self.ch_hi, self.ch_vi = ch_start.ch_h, ch_start.ch_v
        self.ch_hf, self.ch_vf = ch_end.ch_h, ch_end.ch_v

        self.ch_h = self.ch_hi
        self.ch_v = self.ch_vi

        assert(mode in ["linear", "rectangle"])
        self.mode = mode


    def __iter__(self):
        self.ch_h = self.ch_hi
        self.ch_v = self.ch_vi

        return(self)

    def __next__(self):

        # Test if we are on the last iteration
        if self.mode == "linear":
            # Remember to set debug=False so that we don't raise out-of-bounds errors
            # when generating the linear indices
            if(ch_hv_to_num(self.ch_h, self.ch_v, debug=False) > ch_hv_to_num(self.ch_hf, self.ch_vf, debug=False)):
                raise StopIteration

        elif self.mode == "rectangle":
            if((self.ch_h > self.ch_hf) | (self.ch_v > self.ch_vf)):
                raise StopIteration

        # IF we are not out of bounds, make a copy of the current h and v
        ch_h = self.ch_h
        ch_v = self.ch_v

        # Increase the horizontal channel and reset vertical channel number.
        # When iterating linearly, set v to 1
        if self.mode == "linear":
            if(self.ch_v == 8):
                self.ch_v = 1
                self.ch_h += 1
            else:
                self.ch_v += 1

        elif self.mode == "rectangle":
            if(self.ch_v == self.ch_vf):
                self.ch_v = self.ch_vi 
                self.ch_h += 1
            else:
                self.ch_v += 1

        # Return a channel with the previous ch_h 
        return channel(self.dev, ch_h, ch_v)


    @classmethod
    def from_str(cls, range_str):
        """
        Generates a channel_list from a range, specified as
        'ECEI_[LGHT..][0-9]{4}-[0-9]{4}'
        """

        import re

        m = re.search('[A-Z]{1,2}', range_str)
        try:
            dev = m.group(0)
        except:
            raise AttributeError("Could not parse channel string {0:s}".format(ch_str))


        m = re.findall('[0-9]{4}', range_str)

        print(m)

        ch_i = int(m[0])
        ch_vi = ch_i % 100
        ch_hi = int(ch_i // 100)
        ch_i = channel(dev, ch_hi, ch_vi)

        ch_f = int(m[1])
        ch_vf = ch_f % 100
        ch_hf = int(ch_f // 100)
        ch_f = channel(dev, ch_hf, ch_vf)

        return channel_list(ch_i, ch_f)



class channel():
    """Represents an ECEI channel.
    The ECEI array has 24 horizontal channels and 8 vertical channels.

    They are commonly represented as
    L2203
    where L denotes ???, 22 is the horizontal channel and 08 is the vertical channel.
    """

    def __init__(self, dev, ch_h, ch_v):
        """
        Input:
        ======
        dev: string, must be in 'L' 'H' 'G' 'GT' 'GR' 'HR'
        ch_num: int, channel number
        """

        assert(dev in ['L', 'H', 'G', 'HT', 'GR', 'HR'])
        # 24 horizontal channels
        assert((ch_h > 0) & (ch_h  < 25))
        # 8 vertical channels
        assert((ch_v > 0) & (ch_v < 9))

        self.ch_v = ch_v
        self.ch_h = ch_h
        self.dev = dev

        self.ch_num = ch_v * 24 + ch_h

    @classmethod
    def from_str(cls, ch_str):
        """Generates a channel object from a string, such as L2204 or GT1606."""
        import re
        m = re.search('[A-Z]{1,2}', ch_str)
        try:
            dev = m.group(0)
        except:
            raise AttributeError("Could not parse channel string {0:s}".format(ch_str))

        m = re.search('[0-9]{4}', ch_str)
        ch_num = int(m.group(0))
        
        ch_v = (ch_num % 100)
        ch_h = int(ch_num // 100)

        channel1 = cls(dev, ch_h, ch_v)

        return(channel1)

    def __str__(self):
        """Returns a standardized string"""
        ch_str = "{0:s}{1:02d}{2:02d}".format(self.dev, self.ch_h, self.ch_v)

        return(ch_str)

    def idx(self):
        """Returns the linear index corresponding to ch_h and ch_v"""
        return ch_hv_to_num(self.ch_h, self.ch_v)


# End of file channels.py