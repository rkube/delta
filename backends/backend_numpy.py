# Encoding: UTF-8 -*-

from os.path import join
import numpy as np
import logging
import string
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


    def store(self, fname, data, metadata):
        """Stores data and args in numpy file

        Input:
        ======
        fname, string: Filename the data is stored in
        data, ndarray: Data to store in file
        metadata, dict: Dictionary that is stored in file
        
        Returns:
        ========
        None
        """
        fname_fq = join(self.datadir, fname) + f"_s{self.ctr:05d}.npz"

        logging.debug(f"Storing data in {fname_fq:s}")
        np.savez(fname_fq, data=data, **metadata)


    def store_metadata(self, task, fname):
        """Stores metadta in an numpy file

        Input:
        ======
        task, a task_spectral object
        fname, string: Filename
        """
        fname_fq = join(self.datadir, fname) + f"_{task.analysis}_metadata.npz"

        logging.debug(f"Storing metadata in {fname_fq:s}")
        np.savez(fname_fq, **metadata)
    

# End of file backend_numpy.py