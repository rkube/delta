#-*- Encoding: UTF-8 -*-

import numpy as np
import h5py

from analysis.ecei_channel import timebase_streaming, ecei_chunk, normalize_mean


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
        # Number of samples in a chunk
        self.chunk_size = cfg["source"]["chunk_size"]
        self.ecei_cfg = cfg["ECEI_cfg"]
        # Total number of chunks
        self.num_chunks = cfg["source"]["num_chunks"]
        self.current_chunk = 0

        self.tnorm = cfg["ECEI_cfg"]["t_norm"]

        # Callable that performs normalization. This is instantiated once data from
        # the time interval tnorm is read in batch_generator
        self.normalize = False

    def cache(self):
        """Pre-loads all data from HDF5, calculates normalization and
        generates a list of arrays."""

        self.cache = np.zeros([self.ch_range.length(), self.chunk_size * self.num_chunks])

        # Cache the data in memory
        with h5py.File(self.filename, "r",) as df:
            for ch in self.ch_range:
                chname_h5 = f"/ECEI/ECEI_{ch}/Voltage" 
                self.cache[ch.idx(), :] = df[chname_h5][:self.chunk_size * self.num_chunks].astype(np.float64)
        self.cache = self.cache * 1e-4
        

    def batch_generator(self):
        """Loads the next time-chunk from the data file.
        This implementation works as a generator.
        
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
        chunk : ecei_chunk, data from current time chumk
        """

        # Generate start/stop time for timebase
        f_sample = self.ecei_cfg["SampleRate"] * 1e3
        dt = 1. / f_sample
        t_start = self.ecei_cfg["TriggerTime"][0]
        t_end = min(self.ecei_cfg["TriggerTime"][1], t_start + 5_000_000 * dt)

        for current_chunk in range(self.num_chunks):
            # Generate a time-base for the current chunk
            tb_chunk = timebase_streaming(t_start, t_end, f_sample, self.chunk_size, current_chunk)
            # Load current time-chunk from HDF5 file
            current_chunk = ecei_chunk(self.cache[:, current_chunk * self.chunk_size:(current_chunk + 1) * self.chunk_size],
                                       tb_chunk)

            # Determine whether we need to normalize the data
            tidx_norm = [tb.time_to_idx(t) for t in self.ecei_cfg["t_norm"]]

            if tidx_norm[0] == None:
                continue
            elif tidx_norm[0] is not None:
                data_norm = current_chunk.ecei_data[:, :, tidx_norm[0]:tidx_norm[1]]
                offlev = np.median(data_norm, axis=-1, keepdims=True)
                offstd = data_norm.std(axis=-1, keepdims=True)
                self.normalize = normalize_mean(offlev, offstd)

            self.normalize(current_chunk.ecei_data)

            yield time_chunk





# End of file loader_ecei_v2.py