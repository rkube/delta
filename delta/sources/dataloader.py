# -*- Encoding: UTF-8 -*-

"""General dataloaders."""

import numpy as np
import h5py
import logging

from data_models.kstar_ecei import timebase_streaming, ecei_chunk, channel_range_from_str
from data_models.helpers import normalize_mean


def get_loader(cfg_all):
    """Returns data loader appropriate for the configured diagnostic.

    Args:
        cfg_all (dict):
            Configuration dictionary

    Returns:
        dataloader (dataloader):
            Dataloader object
    """
    if cfg_all["diagnostic"]["name"] == "kstarecei":
        return _loader_ecei(cfg_all)
    elif cfg_all["diagnostic"]["name"] == "nstxgpi":
        return _loader_gpi(cfg_all)
    else:
        raise ValueError("No dataloader for " + cfg_all["diagnostic"]["name"])


class _loader_gpi():
    pass


class _loader_ecei():
    """Loads KSTAR ECEi data time-chunk wise for a specified channel range from an HDF5 file."""

    def __init__(self, cfg_all, cache=True):
        """Initializes KSTAR ECEI dataloader.

        Args:
            cfg_all: (dict):
                Global Delta configuration

        Returns:
            None
        """
        self.ch_range = channel_range_from_str(cfg_all["diagnostic"]["datasource"]
                                               ["channel_range"][0])
        # Create a list of paths in the HDF5 file, corresponding to the specified channels
        self.filename = cfg_all["diagnostic"]["datasource"]["source_file"]
        # Number of samples in a chunk
        self.chunk_size = cfg_all["diagnostic"]["datasource"]["chunk_size"]
        # Total number of chunks
        self.num_chunks = cfg_all["diagnostic"]["datasource"]["num_chunks"]
        self.current_chunk = 0

        if cfg_all["diagnostic"]["datasource"]["datatype"] == "int":
            self.dtype = np.int32
        elif cfg_all["diagnostic"]["datasource"]["datatype"] == "float":
            self.dtype = np.float64

        # Generate start/stop time for timebase
        self.f_sample = cfg_all["diagnostic"]["parameters"]["SampleRate"] * 1e3
        self.dt = 1. / self.f_sample
        self.t_start = cfg_all["diagnostic"]["parameters"]["TriggerTime"][0]
        self.t_end = min(cfg_all["diagnostic"]["parameters"]["TriggerTime"][1],
                         self.t_start + 5_000_000 * self.dt)

        # Callable that performs normalization. This is instantiated once data from
        # the time interval t_norm is read in batch_generator
        self.normalize = None
        self.t_norm = cfg_all["diagnostic"]["parameters"]["t_norm"]

        self.logger = logging.getLogger('simple')

        # Whether we use caching for loading data
        self.is_cached = False
        if cache:
            self.cache()
            self.is_cached = True

    def cache(self):
        """Loads data from HDF5 and fills the cache.

        Returns:
            None
        """
        self.cache = np.zeros([self.ch_range.length(), self.chunk_size * self.num_chunks],
                              dtype=self.dtype)

        assert(self.cache.flags.contiguous)

        # Cache the data in memory
        with h5py.File(self.filename, "r",) as df:
            for ch in self.ch_range:
                chname_h5 = f"/ECEI/ECEI_L{ch.ch_v:02d}{ch.ch_h:02d}/Voltage"
                self.cache[ch.get_idx(), :] =\
                    df[chname_h5][:self.chunk_size * self.num_chunks].astype(self.dtype)
        self.cache = self.cache * 1e-4

    def get_chunk_shape(self):
        """Returns the size of chunks.

        Args:
            None

        Returns:
            chunk_shape (tuple [int, int]):
                (Number of channels, time chunk size)
        """
        return (self.ch_range.length(), self.chunk_size)

    def get_chunk_size_bytes(self):
        """Returns the size of a chunk, in bytes."""
        pass

    def batch_generator(self):
        """Loads the next time-chunk from the data file.

        This implementation works as a generator.

        The ECEI data is usually normalized to a fixed offset, calculated using data
        at the beginning of the stream.

        The time interval where the data we normalize to is taken from is given in
        ECEI_config, t_norm. As long as this data is not seen by the reader, raw
        data is returned.

        Once the data we normalize to is seen, the normalization values are calculated.
        After that, the data from the current and all subsequent chunks is normalized.

        The flag self.is_data_normalized is set to false if raw data is returned.
        It is set to true if normalized data is returned.

        >>> batch_gen = loader.batch_generator()
        >>> for batch in batch_gen():
        >>>    ...

        Returns:
            chunk (ecei_chunk)
                ECEI data from current time chunk, possibly normalized
        """
        # Pre-allocate temp array in case we are running non-cached.
        # See if clause in for-loop below
        if not self.is_cached:
            _chunk_data = np.zeros([self.ch_range.length(), self.chunk_size], dtype=self.dtype)

        for current_chunk in range(self.num_chunks):
            # Generate a time-base for the current chunk
            tb_chunk = timebase_streaming(self.t_start, self.t_end, self.f_sample,
                                          self.chunk_size, current_chunk)

            # Load current time-chunk from HDF5 file
            # IF we are running cached, use the data from cache.
            if self.is_cached:
                _chunk_data = self.cache[:, current_chunk *
                                         self.chunk_size:
                                         (current_chunk + 1) *
                                         self.chunk_size]

            # If we haven't cached, load from HDF5
            else:
                with h5py.File(self.filename, "r",) as df:
                    for ch in self.ch_range:
                        chname_h5 = f"/ECEI/ECEI_{ch}/Voltage"
                        _chunk_data[ch.get_idx(), :] = df[chname_h5][current_chunk *
                                                                     self.chunk_size:
                                                                     (current_chunk + 1) *
                                                                     self.chunk_size]
                _chunk_data = _chunk_data * 1e-4

            current_chunk = ecei_chunk(_chunk_data, tb_chunk)
            # Determine whether we need to normalize the data
            tidx_norm = [tb_chunk.time_to_idx(t) for t in self.t_norm]

            if tidx_norm[0] is not None:
                # TODO: Here we create a normalization object using explicit values for the
                # normalization It may be better to just pass the data and let the normalization
                # object figure out how to calculate the needed constants. This would be the best
                # way to allow different normalization.
                data_norm = current_chunk.data()[:, tidx_norm[0]:tidx_norm[1]]
                offlev = np.median(data_norm, axis=-1, keepdims=True)
                offstd = data_norm.std(axis=-1, keepdims=True)
                self.normalize = normalize_mean(offlev, offstd)

                self.logger.info(f"Calculated normalization using\
                                 {tidx_norm[1] - tidx_norm[0]} samples ")

            if self.normalize is not None:
                self.logger.info("Normalizing current_chunk")
                self.normalize(current_chunk.data())
            else:
                self.logger.info("dropping current_chunk: self.normalize has not been initialized")
                continue

            yield current_chunk


# End of file loader_ecei_v2.py
