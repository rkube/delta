# -*- Encoding: UTF-8 -*-

"""Helper functions for managing ADIOS2

Provide channel names etc
"""


def gen_io_name(shotnr: int):
    """Generates an IO name for ADIOS2 objects"""
    return f"KSTAR_ECEI_{shotnr:03d}"

def gen_channel_name(channel_id: int, rank: int):
    """Generates a channel ID for readers."""
    return f"ch{channel_id:04d}_r{rank:03d}"

# End of file adios_helpers.py