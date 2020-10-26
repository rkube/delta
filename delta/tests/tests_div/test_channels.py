# Encoding: UTF-8 -*-

#import sys
#sys.path.append("/home/rkube/repos/delta")
import itertools
import numpy as np


from .context.data_models.channels_2d import num_to_vh, vh_to_num, channel_2d, channel_pair, channel_range, unique_everseen

# Test the correct mapping from channel linear index 0..191 to VV and HH tuple

def test_answer():
    print("Check that the mapping from v and h to a linear channel number is correct")
    a_vh_to_num = vh_to_num(24, 8, 'horizontal')
    a_num_to_vh = num_to_vh(24, 8, 'horizontal')
    ch_idx = 1
    for vv in range(1, 25):
        for hh in range(1, 9):
            assert(a_num_to_vh(ch_idx) == (vv, hh))
            assert(a_vh_to_num(vv, hh) == ch_idx)

            ch_idx += 1

    assert(ch_idx == 193)


    # Deprecated, this is only for KSTAR ECEI
    #print("Testing channel.from_str()")
    #for _ in range(50):
    #     vv = np.random.randint(1, 25)
    #     hh = np.random.randint(1, 9)
    #     ch_str = f"{dd:s}{vv:02d}{hh:02d}"
    #
    #     assert(channels.channel.from_str(ch_str) == channels.channel(dd, vv, hh))
    #
    # print("Testing serialization")
    # for _ in range(50):
    #     dd_list = ['L', 'H', 'G', 'HT', 'GR', 'HR']
    #     dd = dd_list[np.random.randint(0, len(dd_list))]
    #     vv = np.random.randint(1, 25)
    #     hh = np.random.randint(1, 9)
    #
    #     ch = channels.channel(dd, vv, hh)
    #     assert(ch == channels.channel.from_json(ch.to_json()))
    #
    #
    print("Testing equality functions of channel_pair")
    ch1 = channel_2d(2, 3, 24, 8, 'horizontal')
    ch2 = channel_2d(4, 6, 24, 8, 'horizontal')

    assert( ch1 != ch2 )

    assert(channel_pair(ch1, ch2) == channel_pair(ch2, ch1))
    #
    # print("Testing channel pair serialization")
    #
    # for _ in range(50):
    #     dd_list = ['L', 'H', 'G', 'HT', 'GR', 'HR']
    #     dd = dd_list[np.random.randint(0, len(dd_list))]
    #
    #     vv = np.random.randint(1, 25)
    #     hh = np.random.randint(1, 9)
    #     ch1 = channels.channel(dd, vv, hh)
    #
    #     vv = np.random.randint(1, 25)
    #     hh = np.random.randint(1, 9)
    #     ch2 = channels.channel(dd, vv, hh)
    #
    #     pair = channels.channel_pair(ch1, ch2)
    #     assert(channels.channel_pair.from_json(pair.to_json()) == pair)
    #
    #
    print("Testing channel_range")
    ch1 = channel_2d(1, 1, 6, 6, 'horizontal')
    ch2 = channel_2d(6, 6, 6, 6, 'horizontal')
    crg1 = channel_range(ch1, ch2)

    # Test iteration over entire index range
    for i, c in enumerate(crg1):
         assert(c.get_idx() == i)

    # Test iteration over small selection
    ch1 = channel_2d(2, 3, 6, 6, 'horizontal')
    ch2 = channel_2d(5, 5, 6, 6, 'horizontal')
    crg2 = channel_range(ch1, ch2)

    for c, match_idx in zip(crg2, [8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28]):
        assert(c.get_idx() == match_idx)


    print("Testing sets of channel pairs")
    print("Note that this test can fail if duplicate pairs are inserted in chpair_list")
    print("upon generation")

    num_unq_pairs = 20
    # Generate random channel pairs
    ch1_list = [channel_2d(vv, hh, 24, 8, 'horizontal') for vv, hh in zip(np.random.randint(1, 25, num_unq_pairs),
                                                                          np.random.randint(1, 9, num_unq_pairs))]
    ch2_list = [channel_2d(vv, hh, 24, 8, 'horizontal') for vv, hh in zip(np.random.randint(1, 25, num_unq_pairs),
                                                                          np.random.randint(1, 9, num_unq_pairs))]
    chpair_list = [channel_pair(ch1, ch2) for ch1, ch2 in zip(ch1_list, ch2_list)]
    assert(len(chpair_list) == num_unq_pairs)
    #
    # Now we add random duplicates
    num_duplicates = 0
    for _ in range(num_duplicates):
        # Select a random channel pair to duplicate. Note that len(chpair_list) increases in each
        # iteration as we append
        chidx = np.random.randint(0, len(chpair_list))
        # Append this channel pair to the list. Take a 50/50 chance to switch the channels too.
        ch2, ch1 = chpair_list[chidx]
        if np.random.randint(0, 2) == True:
            chpair_list.append(channels_2d.channel_pair(ch2, ch1))
        else:
            chpair_list.append(channels_2d.channel_pair(ch1, ch2))

        assert(len(chpair_list) == num_unq_pairs + num_duplicates)

        # Convert chpair_list, which includes 19 duplicates, to a set.
        chpair_set = set(chpair_list)
        assert(len(chpair_set) == num_unq_pairs)

    # Test iteration over channel ranges using unqiue channel pairs
    print("Testing channel_range")
    ch1 = channel_2d(1, 1, 4, 4, 'horizontal')
    ch2 = channel_2d(4, 4, 4, 4, 'horizontal')
    crg1 = channel_range(ch1, ch2)
    crg2 = channel_range(ch1, ch2)

    res = [channel_pair(c1, c2) for c1 in crg1 for c2 in crg2]
    for i in res:
            print(i)
    unq = list(unique_everseen(res))
    print(f"Non-unique: {len(res):d}, unique channel pairs: {len(unq):d}")

    for i in unq:
        print(i)


if __name__ == "__main__":
    test_answer()


# End of file test_channels.py
