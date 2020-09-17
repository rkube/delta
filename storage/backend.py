# Encoding: UTF-8 -*-

"""
Author: Ralph Kube

Defines a basic interface to the backend-storage classes and helper routines
"""

from analysis.channels import channel, channel_pair

class backend():
    def __init__(self):
        pass

    def store(self):
        pass

def serialize_dispatch_seq(dispatch_seq):
    """Serializes the iteration over the channels

    Parameters
    ----------
    dispatch_seq: list of list, the nested channel iteration list, used by task objects.
                  This is a list of lists of channel_pairs:
                  [
                    [pair1_1, pair1_2, ... pair1_N],
                    [pair2_1, pair2_2, ... pair2_N],
                    ...
                    [pairM_1, pairM_2, ... pairM_N]
                  ]
                  
    Returns
    -------
    j, the json string representation serialization of dispatch_seq
        
    The end result will be a JSON nested array with the serialization of each channel pairs,
    i.e. j_str = '[ [ pair1.to_json(), pair2.to_json() ...], [pair1.to_json(), pair1.to_json()...], ...]
    
    This has the advantage that we can store the entire iteration as one item in a json file
    
    To reconstruct the pairs, load 
    >>> j = json.loads(j_str)
    This loads j as a nested list and the channel pairs are available as dictionaries
    >>> type(j[0][0]) 
    dict
    Channel pairs are recovered as
    >>> pair = channels.channel_pair.from_json(json.dumps(j[0][0]))
    >>> print(pair)
    channel_pair: (ch1=L0101, ch2=L0101)

    When reconstructing we have also that

    >>> len(list(task.get_dispatch_sequence())) == len(j["channel_serialization"])
    True

    >>> len(j["channel_serialization"][0]) == task.channel_chunk_size
    True
    """

    # Step 1: Get the list of channel pair chunks from the task object
    chunk_lists = []
    for ch_it in dispatch_seq:
        chunk_lists.append([c for c in ch_it])

    # We now have the list of channel pairs, f.ex.
    # chunk_list[0] = [channel_pair(L0101, L0101), channel_pair(L0102, L0101), ...()]
    # This data is serialized as a json array like this:
    # json_str = "[channel_pair(L0101, L0101).to_json() + ", " + channel_pair(L0102, L0101).to_json(), ...)]"

    j_lists = []
    for sub_list in chunk_lists:
        j_lists.append("["  + ", ".join([c.to_json() for c in sub_list]) + "]")
    j_str = "[" + ", ".join(j_lists) + "]"

    return j_str

    # Put the channel serialization in the corresponding key
    #j_str = '{"channel_serialization": ' + j_str + '}'
    #
    #
    #j = json.loads(j_str)


def deserialize_dispatch_seq(channel_ser):
    """Returns a list of list of channel pairs, as created by serialize dispatch_seq

    Parameters:
    ===========
    channel_ser: JSON serialization of the channel pairs

    Returns:
    ========
    List of list of channel pairs
    """

    dispatch_seq = []
    
    for pair_list in channel_ser:
        new_list = []
        for pair in pair_list:
            ch1 = channel(pair["ch1"]["dev"], 
                          pair["ch1"]["ch_v"], 
                          pair["ch1"]["ch_h"])
            ch2 = channel(pair["ch2"]["dev"], 
                          pair["ch2"]["ch_v"], 
                          pair["ch2"]["ch_h"])
            new_list.append(channel_pair(ch1, ch2))
        dispatch_seq.append(new_list)

    return dispatch_seq


# End of file backend.py