# Encoding: UTF-8 

"""
Find out how to get a channel list from tget_dispatch_sequence
and pickle it
"""


#import sys
#sys.path.append("/global/homes/r/rkube/repos/delta")
import more_itertools
from itertools import chain

from context.data_models.channels_2d import channel_2d, channel_range, channel_pair


ref_channels = channel_range.from_str("L0101-2408")
cmp_channels = channel_range.from_str("L0101-2408")
channel_pairs = [channel_pair(ref, cmp) for ref in ref_channels for cmp in cmp_channels]
unique_channels = list(more_itertools.distinct_combinations(channel_pairs, 1))


print("Number of channel pairs: {0:d}".format(len(channel_pairs)))
print("Unique channel paits: {0:d}".format(len(unique_channels)))

ch_list = [u[0] for u in unique_channels]

#np.savez("dispatch_seq.npz", ch_list=ch_list)

# End of file test_channel_it.py