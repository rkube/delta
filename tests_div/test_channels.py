# Encoding: UTF-8 -*-

import sys
sys.path.append("/global/homes/r/rkube/repos/delta")
import itertools


from analysis.channels import channel, channel_range
ch_start = channel("L", 1, 3)
ch_end = channel("L", 4, 5)

# We often need a rectangular selection. This mode refers to the rectangular configuration
# of the channel views. Basically, we specify a recatngle by giving a lower left corner 
# and an upper right corner
# Then iterate within the rectangular bounds defined by this rectangle.
clist = (ch_start, ch_end, mode="rectangle")
for c in clist:
    print(c)

# Test naive iteration where we just take the outer product of two lists:
crg1 = channel_range.from_str("L0101-0104")
crg2 = channel_range.from_str("L0201-0204")

print("Testing naive iteration...")
res = ["{0:d} x {1:d}".format(c1.idx(), c2.idx()) for c1 in clist1 for c2 in clist2]
print(res)


# Test iteration over groups of channels.
# In the spectral analysis we often want to iterate over a list of unique channel pairs.
# That is, the pair (ch1, ch2) is identical to the pair (ch2, ch1).
# We can do this using combinations_with_replacement from itertools as follows.
#
print("Testing iteration over channel lists...")
crg1 = channel_range.from_str("L0101-0104")
combs = list(itertools.combinations_with_replacement(crg1, 2))
for c in combs:
    print(c[0], c[1])


# We also have the situation where we have a list of reference channels and a list of cross channels.
# This situation can be accomodated by 'merging' iterators over the two channel lists and then using
# the method above on the resulting channel list.
print("Testing iteration over two channel lists...")

crg_total = itertools.chain(crg1, crg2)
combs = list(itertools.combinations_with_replacement(crg_total, 2))

for c in combs:
    print(c[0], c[1])



# End of file test_channels.py