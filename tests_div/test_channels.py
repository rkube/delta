# Encoding: UTF-8 -*-

import sys
sys.path.append("/global/homes/r/rkube/repos/delta")
import itertools
import numpy as np


from analysis import channels #.channels import channel, channel_range, channel_pair, unique_everseen

# Test the correct mapping from channel linear index 0..191 to VV and HH tuple

def test_answer():
    print("Check that the mapping from v and h to a linear channel number is correct")
    ch_idx = 1
    for vv in range(1, 25):
        for hh in range(1, 9):
            assert(channels.ch_num_to_vh(ch_idx) == (vv, hh))
            assert(channels.ch_vh_to_num(vv, hh) == ch_idx)

            ch_idx += 1

    assert(ch_idx == 193)
        

    print("Testing channel.from_str()")
    for _ in range(50):
        dd_list = ['L', 'H', 'G', 'HT', 'GR', 'HR']
        dd = dd_list[np.random.randint(0, len(dd_list))]
        vv = np.random.randint(1, 25)
        hh = np.random.randint(1, 9)
        ch_str = f"{dd:s}{vv:02d}{hh:02d}"

        assert(channels.channel.from_str(ch_str) == channels.channel(dd, vv, hh))


    print("Testing equality functions of channel_pair")
    ch1 = channels.channel("G", 2, 3)
    ch2 = channels.channel("G", 4, 6)

    assert(channels.channel_pair(ch1, ch2) == channels.channel_pair(ch2, ch1))
    assert(hash(channels.channel_pair(ch1, ch2)) == hash(channels.channel_pair(ch2, ch1)))


    print("Testing channel_range")
    print("Linear selection:")   

    vv, hh = channels.ch_num_to_vh(1)
    ch1 = channels.channel("L", vv, hh)
    vv, hh = channels.ch_num_to_vh(192)
    ch2 = channels.channel("L", vv, hh)

    crg = channels.channel_range(ch1, ch2, "linear")

    i = 1
    for c in crg:
        assert(c.idx() == i)
        i -= -1


    print("Testing sets of channel pairs")
    print("Note that this test can fail if duplicate pairs are inserted in chpair_list")
    print("upon generation")

    num_unq_pairs = 20
    # Generate random channel pairs
    ch1_list = [channels.channel("L", vv, hh) for vv, hh in zip(np.random.randint(1, 25, num_unq_pairs),
                                                                np.random.randint(1, 9, num_unq_pairs))]

    ch2_list = [channels.channel("L", vv, hh) for vv, hh in zip(np.random.randint(1, 25, num_unq_pairs),
                                                                np.random.randint(1, 9, num_unq_pairs))]                                                                
    chpair_list = [channels.channel_pair(ch1, ch2) for ch1, ch2 in zip(ch1_list, ch2_list)]
    assert(len(chpair_list) == num_unq_pairs)

    # Now we add random duplicates
    num_duplicates = 0
    for _ in range(num_duplicates):
        # Select a random channel pair to duplicate. Note that len(chpair_list) increases in each
        # iteration as we append
        chidx = np.random.randint(0, len(chpair_list))
        # Append this channel pair to the list. Take a 50/50 chance to switch the channels too.
        ch2, ch1 = chpair_list[chidx]
        if np.random.randint(0, 2) == True:
            chpair_list.append(channels.channel_pair(ch2, ch1))
        else:
            chpair_list.append(channels.channel_pair(ch1, ch2))

    assert(len(chpair_list) == num_unq_pairs + num_duplicates)

    # Convert chpair_list, which includes 19 duplicates, to a set.
    chpair_set = set(chpair_list)
    assert(len(chpair_set) == num_unq_pairs)


# # Test iteration over channel ranges using unqiue channel pairs

# res = [channel_pair(c1, c2) for c1 in crg1 for c2 in crg2]
# # res contains duplicate paris, f.ex. (L0202, L0203) and (L0203, L0202)
# unq = unique_everseen(res)
# unq_list = list(unq)

# print("Non-unique: {0:d}, unique channel pairs: {1:d}".format(len(res), len(unq_list)))




# End of file test_channels.py