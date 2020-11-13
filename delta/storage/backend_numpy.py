# Encoding: UTF-8 -*-

"""Numpy storage backend."""


from os.path import join
import numpy as np
import logging
import json


from storage.helpers import serialize_dispatch_seq


class backend_numpy():
    """Storage class that stores analysis results in numpy arrays."""
    def __init__(self, cfg):
        """Initializes the class.

        Args:
            cfg (dict):
                config.storage part of the Delta config object.

        Returns:
            None
        """
        super().__init__()

        # Directory where numpy files are stored
        self.basedir = cfg['basedir']

    def store(self, chunk_data, chunk_info):
        """Stores data and args in numpy file.

        Args:
            chunk_data (ndarray):
                Data to store in file
            chunk_info (dict):
                Info dictionary returned from the future

        Returns:
            None
        """
        fname_fq = join(self.basedir, chunk_info['analysis_name']) +\
            f"_tidx{chunk_info['tidx']:05d}_batch{chunk_info['channel_batch']:02d}.npz"

        logging.debug("Storing data in " + fname_fq)
        np.savez(fname_fq, chunk_data, analysis_name=chunk_info['analysis_name'],
                 tidx=chunk_info['tidx'], batch=chunk_info['channel_batch'])

    def store_metadata(self, cfg, dispatch_seq):
        """Stores metadta in an numpy file.

        The dispatch sequence from a task object is returned by task.get_dispatch_sequence()

        Args:
            cfg (dict):
                the json configuration passed into the processor...
            dispatch_seq (iterable):
                The dispatch sequence from task.

        Returns:
            None

        """
        logging.debug("Storing metadata in " + self.basedir)

        # # Step 1: Get the list of channel pair chunks from the task object
        # chunk_lists = []
        # for ch_it in task.get_dispatch_sequence():
        #     chunk_lists.append([c for c in ch_it])

        # # We now have the list of channel pairs, f.ex.
        # # chunk_list[0] = [channel_pair(L0101, L0101), channel_pair(L0102, L0101), ...()]
        # # This data is serialized as a json array like this:
        # # json_str = "[channel_pair(L0101, L0101).to_json() + ", "
        # # + channel_pair(L0102, L0101).to_json(), ...)]"

        # j_lists = []
        # for sub_list in chunk_lists:
        #     j_lists.append("["  + ", ".join([c.to_json() for c in sub_list]) + "]")
        # j_str = "[" + ", ".join(j_lists) + "]"

        j_str = serialize_dispatch_seq()
        # Put the channel serialization in the corresponding key
        j_str = '{"channel_serialization": ' + j_str + '}'
        j = json.loads(j_str)
        # Adds the channel_serialization key to cfg
        cfg.update(j)

        with open(join(self.basedir, "config.json"), "w") as df:
            json.dump(cfg, df)

        return None

# End of file backend_numpy.py
