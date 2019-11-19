# Encoding: UTF-8 -*-

import sys
sys.path.append("/global/homes/r/rkube/repos/delta")

from analysis.channels import channel, channel_list
ch_start = channel("L", 1, 2)
ch_end = channel("L", 12, 4)


clist = channel_list(ch_start, ch_end, mode="rectangle")

for c in clist:
    print(c)


# End of file test_channels.py