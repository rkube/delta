#-*- Coding: UTF-8 -*-

import h5py
#from mpi4py import MPI
import numpy as np

from analysis.channels import channel_range

class loader_ecei():
    """Loads KSTAR ECEi data time-chunk wise for a specified channel range from an HDF5 file"""


    def __init__(self, cfg):
        """

        Old: filename: str, ch_range: channel_range, chunk_size:int):
        Inputs:
        =======
        cfg: Delta configuration with loader and ECEI section.

        """

        self.ch_range = channel_range.from_str(cfg["source"]["channel_range"][0])
        # Create a list of paths in the HDF5 file, corresponding to the specified channels
        self.filename = cfg["source"]["source_file"]
        self.chunk_size = cfg["source"]["chunk_size"]
        self.ecei_cfg = cfg["ECEI_cfg"]
        self.num_chunks = cfg["source"]["num_chunks"]
        self.current_chunk = 0

        self.tnorm = cfg["ECEI_cfg"]["t_norm"]

        self.got_normalization = False

        self.chunk_list = None

    def gen_timebase(self, chunk_num):
        """Create a dummy time base for chunk last read.

        Parameters:
        ===========
        chunk_num: Number of the chunk to generate a time-base for
        """

        # Unpack the trigger time, plasma time and end time from TriggerTime
        tt0, pl, tt1 = self.ecei_cfg["TriggerTime"]
        # Get the sampling frequency, in units of Hz
        fs = self.ecei_cfg["SampleRate"] * 1e3

        # The time base starts at t0. Assume samples are streaming in continuously with sampling frequency fs.
        # Offset of the first sample in current chunk
        offset = self.chunk_size * chunk_num
        t_sample = 1. / fs
        # Use integers in arange to avoid round-off errors. We want exactly chunk_size elements in this array
        tb = np.arange(offset, offset + self.chunk_size, dtype=np.float64)
        # Scale timebase with t_sample and shift to tt0
        tb = (tb * t_sample) + tt0        
        return(tb)

    
    def cache(self):
        """Pre-loads all data from HDF5, calculates normalization and
        generates a list of arrays."""

        self.cache = np.zeros([self.ch_range.length(), self.chunk_size * self.num_chunks])

        # Cache the data in memory
        with h5py.File(self.filename, "r",) as df:
            for ch in self.ch_range:
                chname_h5 = f"/ECEI/ECEI_{ch}/Voltage" 
                self.cache[ch.idx(), :] = df[chname_h5][:self.chunk_size * self.num_chunks].astype(np.float64)
        

    def batch_generator(self):
        """Loads the next time-chunk from the data file.
        This implementation works as a generator.
        In production use this as
        
        The ECEI data is usually normalized to a fixed offset, calculated using data 
        at the beginning of the stream.

        The time interval where the data we normalize to is taken from is given in ECEI_config, t_norm.
        As long as this data is not seen by the reader, raw data is returned.
        
        Once the data we normalize to is seen, the normalization values are calculated.
        After that, the data from the current and all subsequent chunks is normalized.
    
        The flag self.is_data_normalized is set to false if raw data is returned.
        It is set to true if normalized data is returned.

        >>> batch_gen = loader.batch_generator()
        >>> for batch in batch_gen():
        >>>    ...

        yields
        =======
        time_chunk : ndarray, data from current time chumk
        """

        for current_chunk in range(self.num_chunks):
            # Load current time-chunk from HDF5 file
            time_chunk = self.cache[:, current_chunk * self.chunk_size:(current_chunk + 1) * self.chunk_size]

            # Scale, see get_data in kstarecei.py
            time_chunk = time_chunk * 1e-4

            # See if we can calculate the normalization.
            if self.got_normalization:
                # This corresponds to norm == 1 in kstarecei.py
                time_chunk = (time_chunk - self.offset_lvl) / time_chunk.mean(axis=1, keepdims=True) - 1.0
            elif self.got_normalization == False:
                tb = self.gen_timebase(current_chunk)
                tnorm_idx = (tb > self.tnorm[0]) & (tb < self.tnorm[1])

                if tnorm_idx.sum() > 100:
                    self.offset_lvl = np.median(time_chunk[:, tnorm_idx], axis=1, keepdims=True)
                    self.offset_std = time_chunk[:, tnorm_idx].std(axis=1)
                    self.got_normalization = True


            yield time_chunk


# End of file 