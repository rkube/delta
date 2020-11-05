# -*- Encoding: UTF-8 -*-


class channel_2d:
    """Abstraction of a channel in 2d array"""

    def __init__(self, ch_v, ch_h, chnum_v, chnum_h, order):
        """Initializes channel_2d class

        Args:
            ch_v (int):
              Horizontal channel number
            ch_h (int):
              Vertical channel number
            chnum_v (int):
              Total count of vertical channels
            chnum_h (int):
              Total count of vertical channels
            order (string):
              Either 'horizontal' or 'vertical'. Denotes whether horizontal or
              vertical channels are arranged consecutively
        """
        # horizontal channels
        assert((ch_v > 0) & (ch_v <= chnum_v))
        # vertical channels
        assert((ch_h > 0) & (ch_h <= chnum_h))
        #
        assert(order in ['horizontal', 'vertical'])

        self.ch_v, self.ch_h = ch_v, ch_h
        self.chnum_v, self.chnum_h = chnum_v, chnum_h
        # Index functor
        self.idx_fct = vh_to_num(self.chnum_v, self.chnum_h, order)
        self.order = order

    def __str__(self):
        return f"({self.ch_v:03d}, {self.ch_h:03d})"

    def __eq__(self, other):
        """Define equality for two channels when both ch_h, and ch_v are equal to one another."""
        return (self.ch_v == other.ch_v) and (self.ch_h == other.ch_h)

    def get_num(self):
        return self.idx_fct(self.ch_v, self.ch_h)

    def get_idx(self):
        """Returns the linear, ZERO-BASED, index corresponding to ch_h and ch_v

        Returns:
            index (int):
               Linear, zero-based index corresponding to ch_h and ch_v in the 2d array.
        """
        return self.idx_fct(self.ch_v, self.ch_h) - 1


class channel_pair:
    """Custom defined channel pair.

    This is just a tuple with an overloaded equality operator. It's also hashable so one can
    use it in conjunction with sets.

    >>> ch1 = channel(13, 7, 24, 8, 'horizontal')
    >>> ch2 = channel(12, 7, 24, 8, 'horizontal')
    >>> ch_pair = channel_pair(ch1, c2)

    >>> channel_pair(ch1, ch2) == channel_pair(ch2, c1)
    True

    The hash is that of a tuple consisting of the ordered channel indices of ch1 and ch2.
    """

    def __init__(self, ch1, ch2):
        """Initializes channel pair with 2 channels.

        Args:
            ch1 (:py:class:`channel_2d`):
                First channel
            ch2 (:py:class:`channel_2d`):
                Second channel

        """
        self.ch1 = ch1
        self.ch2 = ch2

    def __eq__(self, other):
        return(((self.ch1 == other.ch1) and (self.ch2 == other.ch2)) or
               ((self.ch1 == other.ch2) and (self.ch2 == other.ch1)))

    def __str__(self):
        """Returns a standardized string"""

        ch_str = f"{self.__class__.__name__}: (ch1={self.ch1}, ch2={self.ch2})"
        return(ch_str)

    def __iter__(self):
        yield from [self.ch1, self.ch2]

    def __hash__(self):
        """Implement hash so that we can use sets."""

        return hash((min(self.ch1.get_idx(), self.ch2.get_idx()),
                     max(self.ch1.get_idx(), self.ch2.get_idx())))

    # def to_json(self):
    #     return('{"ch1": ' + self.ch1.to_json() + ', "ch2": ' + self.ch2.to_json() + '}')
    # @classmethod
    # def from_json(cls, str):
    #     j = json.loads(str)
    #     ch1 = channel.from_json(json.dumps(j["ch1"]))
    #     ch2 = channel.from_json(json.dumps(j["ch2"]))
    #     cpair = cls(ch1, ch2)
    #     return cpair


class channel_range:
    """Defines iterators over a 2d sub-array.

       This class defines an iterator over a rectangular selection in a 2d sub-array,
       as defined by vertical and horizontal initial and final position (vi, hi), and (vf, hf).

       .. line-block::
             v
             ^
             |
           6 | oooooo
           5 | ooxxxo
           4 | ooxxxo
           3 | ooxxxo
           2 | ooxxxo
           1 | oooooo
              +--------> h
               123456

     The rectangular selection above shows (vi,hi) = (2,3) and (vf, hf) = (5,5).
     Iteration over this selection with horizontal channels consecutively ordered
     gives the index series

     .. line-block::
         (3,2), (4,2), (5,2),
         (3,3), (4,3), (5,3),
         (3,4), (4,4), (5,4),
         (4,5), (4,5), (5,5).

    """

    def __init__(self, ch_start, ch_end):
        """

        Args:
            ch_start (:py:class:`channel_2d`): Initial channel for iteration
            ch_end (:py:class:`channel_2d`): Stop channel for iteration
        """

        assert(ch_start.get_num() <= ch_end.get_num())
        assert(ch_start.chnum_v == ch_end.chnum_v)
        assert(ch_start.chnum_h == ch_end.chnum_h)
        assert(ch_start.order == ch_end.order)

        self.ch_start = ch_start
        self.ch_end = ch_end
        self.ch_hi, self.ch_vi = ch_start.ch_h, ch_start.ch_v
        self.ch_hf, self.ch_vf = ch_end.ch_h, ch_end.ch_v
        self.order = ch_start.order
        self.chnum_h, self.chnum_v = ch_start.chnum_h, ch_start.chnum_v

        # Set initial position
        self.ch_h = self.ch_hi
        self.ch_v = self.ch_vi

    def __iter__(self):
        """Returns an iterator over the selected range"""
        self.ch_v = self.ch_vi
        self.ch_h = self.ch_h

        return(self)

    def __next__(self):
        """Advances an iterator"""

        # See if we are at the end of the iteration
        if ((self.ch_v > self.ch_vf) | (self.ch_h > self.ch_hf)):
            raise StopIteration

        ch_h, ch_v = self.ch_h, self.ch_v

        # Depending on order, increase ch_h or ch_v. If this channel is at the
        # end of the allowed range, reset it to its initial value and increase
        # ch_v or ch_h (the other one).
        if self.order == 'vertical':
            # If we are at the end of the vertical range, reset vertical position
            # and increase horizontal position
            if self.ch_v == self.ch_vf:
                self.ch_v = self.ch_vi
                self.ch_h += 1
            # If we are not at the end, increase vertical position by one.
            else:
                self.ch_v += 1

        elif self.order == 'horizontal':
            # Same as other if-clause but with ch_v and ch_h flipped
            if self.ch_h == self.ch_hf:
                self.ch_h = self.ch_hi
                self.ch_v += 1
            # If we are not at the end, increase vertical position by one.
            else:
                self.ch_h += 1

        return channel_2d(ch_v, ch_h, self.chnum_v, self.chnum_h, self.order)

    def length(self):
        """Returns the number of channels in the range.

        Returns:
            int: Number of channels in the range
        """
        return(self.ch_end.get_num() - self.ch_start.get_num() + 1)


class num_to_vh():
    """Functor that returns a tuple (ch_v, ch_h) for a channel number.
    Note that these are 1-based numbers.

    >>> obj = num_to_vh(24, 8, "vertical")
    >>> ch_num_to_vh(17)
    (3, 1)
    """

    def __init__(self, chnum_v: int, chnum_h: int, order: str):
        """Initializes with number of vertical and horizontal views.

        Args:
            chnum_v (int): Number of vertical views in the diagnostic
            chnum_h (int): Number of horizontal views in the diagnostic
            order (string) : Either 'horizontal' or 'vertical' denotes whether horizontal
                             or vertical channels are ordered consecutively.
        """

        self.chnum_v = chnum_v
        self.chnum_h = chnum_h
        self.order = order

    def __call__(self, ch_num):
        """Converts 2d indices ch_v and ch_h to linear index.

        Args:
            ch_num (int): Linear, one-based channel number

        Returns:
            (ch_v, ch_h) (tuple): Vertical and horizontal channel view
        """
        assert((ch_num >= 1) & (ch_num <= self.chnum_v * self.chnum_h))
        # Calculate using zero-base
        ch_num -= 1
        ch_v = ch_num // self.chnum_h
        ch_h = ch_num - ch_v * self.chnum_h
        return (ch_v + 1, ch_h + 1)


class vh_to_num:
    """Returns the linear channel number for a tuple of channel indices.

    >>> obj = vh_to_num(24, 8, order='horizontal')
    >>> obj(2, 4)
    12

    >>> obj = vh_to_num(24, 8, order='vertical')
    >>> obj(2, 4)
    28
    """

    def __init__(self, chnum_v, chnum_h, order="horizontal"):
        """ Initializes the functor class.

        Args:
            ch_v (int):
                vertical channel number
            ch_h (int):
                horizontal channel number
            order (str) :
                Either 'horizontal' or 'vertical'. Identifies the whether the horizontal
                or vertical ordered consecutively.

        """

        assert(order in ['horizontal', 'vertical'])

        self.chnum_v = chnum_v
        self.chnum_h = chnum_h
        self.order = order

    def __call__(self, ch_v, ch_h):
        """Converts ch_v and ch_h to linear index.

        Args:
            (ch_v, ch_h) (tuple):
                Vertical and horizontal channel view

        Returns:
            ch_num (int):
                Linear, one-based channel number
        """

    # We usually want to check that we are within the bounds.
    # But sometimes it is helpful to avoid this.
    # if debug:
        assert((ch_v > 0) & (ch_v < self.chnum_v + 1))
        assert((ch_h > 0) & (ch_h < self.chnum_h + 1))

        return((ch_v - 1) * self.chnum_h + ch_h)

# End of file channels_2d.py
