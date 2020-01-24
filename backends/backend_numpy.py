# Encoding: UTF-8 -*-

from os.path import join
import numpy as np
import logging
import json
#from backends.backend import backend

from .backend import backend, serialize_dispatch_seq


class backend_numpy(backend):
    """
    Author: Ralph Kube
    
    Defines a method to store results from a task in numpy arrays."""
    def __init__(self, cfg):
        """
        Inputs
        ======
        datadir, str: Base directory where data is stored
        """
        super().__init__()

        # Directory where numpy files are stored
        print(cfg)
        self.basedir = cfg['basedir']


    def store(self, chunk_data, chunk_info):
        """Stores data and args in numpy file

        Input:
        ======
        chunk_data, ndarray: Data to store in file
        chunk_info, dict: Info dictionary returned from the future
        
        Returns:
        ========
        None
        """

        fname_fq = join(self.basedir, chunk_info['analysis_name']) + f"_tidx{chunk_info['tidx']:05d}_batch{chunk_info['channel_batch']:02d}.npz"

        logging.debug("Storing data in " + fname_fq)
        np.savez(fname_fq, chunk_data, analysis_name=chunk_info['analysis_name'],
                 tidx=chunk_info['tidx'], batch=chunk_info['channel_batch'])


    def store_metadata(self, cfg, dispatch_seq):
        """Stores metadta in an numpy file

        Parameters
        ----------
        cfg, the json configuration passed into the processor...
        dispatch_seq: The dispatch sequence from task.


        The dispatch sequence from a task object is returned by task.get_dispatch_sequence()
        """

        logging.debug("Storing metadata in " + self.basedir)

        # # Step 1: Get the list of channel pair chunks from the task object
        # chunk_lists = []
        # for ch_it in task.get_dispatch_sequence():
        #     chunk_lists.append([c for c in ch_it])

        # # We now have the list of channel pairs, f.ex.
        # # chunk_list[0] = [channel_pair(L0101, L0101), channel_pair(L0102, L0101), ...()]
        # # This data is serialized as a json array like this:
        # # json_str = "[channel_pair(L0101, L0101).to_json() + ", " + channel_pair(L0102, L0101).to_json(), ...)]"

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
    

# End of file backend_numpy.py