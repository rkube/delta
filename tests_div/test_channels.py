# Encoding: UTF-8 -*-

import sys
sys.path.append("/global/homes/r/rkube/repos/delta")

from analysis.channels import channel, channel_list
ch_start = channel("L", 1, 3)
ch_end = channel("L", 4, 5)


clist = channel_list(ch_start, ch_end, mode="rectangle")

for c in clist:
    print(c)


clist1 = channel_list.from_str("L0101-0108")
clist2 = channel_list.from_str("L0301-0308")

res = ["{0:d} x {1:d}".format(c1.idx(), c2.idx()) for c1 in clist1 for c2 in clist2]
print(res)



# End of file test_channels.py