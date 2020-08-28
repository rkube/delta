# -*- Construct an ECEI view made up of ecei_channels


import numpy as numpy
from ecei_channel import ecei_channel


class ecei_view():
    """Defines the view of an ECEI"""

    def __init__(self, num_v=24, num_h=8):
        # Initializes the ECEI view with zero.
        self.init = False

        self.low_signals = np.zeros([num_v, num_h], dtype=bool)
        


    def 


# End of file ecei_view