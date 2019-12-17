# Encoding: UTF-8 -*-

from os.path import join
import numpy as np
from backends.backend import backend


class backend_numpy(backend):
    def __init__(self, datadir):
        """
        Inputs
        ======
        datadir, str: Base directory where data is stored
        """

        super().__init__()

        self.datadir = datadir
        # Counter for input files
        self.ctr = 0

    def store(self, fname, data):
        """Stores data and args in numpy file

        Input:
        ======
        fname, string: Filename the data is stored in
        data, ndarray: Data to story in
        *args, dict: descriptive meta-data
        
        Returns:
        ========
        None
        """
        fname_fq = join(self.datadir, fname) + "_s{0:05d}".format(self.ctr) + ".npz"

        print("Storing data in " + fname_fq)
        np.savez(fname_fq, data=data)
    





# End of file backend_numpy.py