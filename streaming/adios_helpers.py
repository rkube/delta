# -*- Encoding: UTF-8 -*-

"""

Author: Ralph Kube

Helper functions for managing ADIOS2
"""


def gen_io_name(shotnr: int):
    """Generates an IO name for ADIOS2 objects"""

    return f"KSTAR_ECEI_{shotnr:03d}"

def gen_channel_name(channel_id: int, rank: int):
    """Generates a channel ID for readers."""

    return f"ch{channel_id:04d}_r{rank:03d}"

def gen_channel_name_v2(shotnr: int, channel_rg: str):
    """Generates a channel ID using channel range strings. (see analysis/channels.py)"""

    return f"{shotnr:05d}_ch{channel_rg:s}"


def gen_channel_name_v3(prefix: str, shotnr: int, channel_rg: str):

    return f"{prefix}/{shotnr:05d}_ch{channel_rg:s}"


# End of file adios_helpers.py