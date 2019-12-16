# Encoding: UTF-8 -*-

import sys
sys.path.append("/global/homes/r/rkube/repos/delta")
import itertools


from analysis.channels import channel, channel_range, channel_pair, unique_everseen
ch_start = channel("L", 1, 3)
ch_end = channel("L", 4, 5)

# We often need a rectangular selection. This mode refers to the rectangular configuration
# of the channel views. Basically, we specify a recatngle by giving a lower left corner 
# and an upper right corner
# Then iterate within the rectangular bounds defined by this rectangle.
clist = channel_range(ch_start, ch_end, mode="rectangle")
for c in clist:
    print(c)

# Test naive iteration where we just take the outer product of two lists:
crg1 = channel_range.from_str("L0201-0303")
crg2 = channel_range.from_str("L0202-0304")

print("Testing naive iteration...")
#res = ["{0:d} x {1:d}".format(c1.idx(), c2.idx()) for c1 in crg1 for c2 in crg2]
res = ["{0:s} x {1:s}".format(c1.__str__(), c2.__str__()) for c1 in crg1 for c2 in crg2]
print(len(res))
print(res)


# Test iteration over channel ranges using unqiue channel pairs

res = [channel_pair(c1, c2) for c1 in crg1 for c2 in crg2]
# res contains duplicate paris, f.ex. (L0202, L0203) and (L0203, L0202)
unq = unique_everseen(res)
unq_list = list(unq)

print("Non-unique: {0:d}, unique channel pairs: {1:d}".format(len(res), len(unq_list)))




# End of file test_channels.py