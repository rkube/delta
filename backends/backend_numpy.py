# Encoding: UTF-8 -*-

from os.path import join
import numpy as np
import logging
import json
#from backends.backend import backend

from .backend import backend


class backend_numpy(backend):
    """
    Author: Ralph Kube
    
    Defines a method to store results from a task in numpy arrays."""
    def __init__(self, datadir):
        """
        Inputs
        ======
        datadir, str: Base directory where data is stored
        """
        super().__init__()

        # Directory where numpy files are stored
        self.datadir = datadir
        # Counter for input files
        self.ctr = 0


    def store(self, cfg, chunk_data, chunk_info):
        """Stores data and args in numpy file

        Input:
        ======
        fname, string: Filename the data is stored in
        cfg, dict: Configuration of the run
        data, ndarray: Data to store in file
        metadata, dict: Info dictionary returned from the future
        
        Returns:
        ========
        None
        """
        fname_fq = join(cfg['basedir'], fname) + f"_s{self.ctr:05d}.npz"

        logging.debug(f"Storing data in {fname_fq:s}")
        np.savez(fname_fq, data=data, **metadata)


    def store_metadata(self, cfg, task):
        """Stores metadta in an numpy file

        Parameters
        ----------
        cfg, the json configuration passed into the processor...
        task, a task_spectral object

        Serialize the iteration over the channels
        =========================================
        
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
        #fname_fq = join(self.datadir, fname) + f"_{task.analysis}_metadata.npz"

        assert('basedir' in cfg.keys())

        logging.debug(f"Storing metadata in {cfg['basedir']:s}")

        # Step 1: Get the list of channel pair chunks from the task object
        chunk_lists = []
        for ch_it in task.get_dispatch_sequence():
            chunk_lists.append([c for c in ch_it])

        # We now have the list of channel pairs, f.ex.
        # chunk_list[0] = [channel_pair(L0101, L0101), channel_pair(L0102, L0101), ...()]
        # This data is serialized as a json array like this:
        # json_str = "[channel_pair(L0101, L0101).to_json() + ", " + channel_pair(L0102, L0101).to_json(), ...)]"

        j_lists = []
        for sub_list in chunk_lists:
            j_lists.append("["  + ", ".join([c.to_json() for c in sub_list]) + "]")
        j_str = "[" + ", ".join(j_lists) + "]"
        
        # Put the channel serialization in the corresponding key
        j_str = '{"channel_serialization": ' + j_str + '}'

        j = json.loads(j_str)

        # Adds the channel_serialization key to cfg
        cfg.update(j)
       
        with open(join(cfg['basedir'], "config.json"), "w") as df:
            json.dump(cfg, df)
    

# End of file backend_numpy.py