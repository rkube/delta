# -*- Encoding: UTF-8 -*-

"""General dataloaders."""

import os
import numpy as np
import h5py
import logging
import re

from data_models.kstar_ecei import ecei_chunk, channel_range_from_str
from data_models.timebase import timebase_streaming


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

        # Process attributes stored in HDF5 file
        # Extract device name of filename, the part in between .[A-Z]{2}.
        m = re.search('\.[A-Z]{1,2}\.', os.path.basename(self.filename))
        self.attrs = {"dev": m[0][1:-1]}
        self._read_attributes_from_hdf5()
        self.logger = logging.getLogger('simple')

        # Whether we use caching for loading data
        self.is_cached = False
        if cache:
            self.cache()
            self.is_cached = True

    def _read_attributes_from_hdf5(self):
        """Reads attributes from HDF5.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        with h5py.File(self.filename, 'r') as df:
            dset = df["ECEI"].attrs
            for attr_name in ["SampleRate", "TriggerTime", "TFcurrent", "Mode", "LoFreq",
                              "LensFocus", "LensZoom"]:
                try:
                    # Test if attribute is serializable. 
                    # If it is f.ex an ndarray convert it to a list.
                    new_attr = dset[attr_name]
                    if isinstance(new_attr, np.ndarray):
                        new_attr = list(new_attr)
                    if isinstance(new_attr, bytes):
                        new_attr = new_attr.strip().decode('utf-8')
                    self.attrs.update({attr_name: new_attr})
                except KeyError:
                    self.logger.info(f"Attribute {attr_name} note found in file {self.filename}")
            # Manually pretti-fy samplerate, tfcurrent, and Mode
            # Store Toroidal Field Coil current in Amps
            self.attrs["TFcurrent"] = self.attrs["TFcurrent"] * 1e3
            # Store SampleRate in Hz
            self.attrs["SampleRate"] = self.attrs["SampleRate"][0] * 1e3
            # self.attrs["Mode"] = self.attrs["Mode"].strip().decode()

    def _read_from_hdf5(self, array, idx_start, idx_end):
        """Reads data from HDF5 into array.

        Values in array are changed in-place.

        Args:
            array (np.ndarray):
                Array where we store HDF5 data
            idx_start (int):
                First index to read
            idx_end (int):
                Last index to read

        Returns:
            None
        """
        # Cache the data in memory
        with h5py.File(self.filename, "r",) as df:
            for ch in self.ch_range:
                chname_h5 = f"/ECEI/ECEI_{self.attrs['dev']}{ch.ch_v:02d}{ch.ch_h:02d}/Voltage"
                array[ch.get_idx(), :] = df[chname_h5][idx_start:idx_end].astype(self.dtype)
        array[:] = array[:] * 1e-4

    def cache(self):
        """Loads data from HDF5 and fills the cache.

        Returns:
            None
        """
        self.cache = np.zeros([self.ch_range.length(), self.chunk_size * self.num_chunks],
                              dtype=self.dtype)
        assert(self.cache.flags.contiguous)

        # Load contents of entire HDF5 file into self.cache
        self._read_from_hdf5(self.cache, 0, self.chunk_size * self.num_chunks)

    def get_chunk_shape(self):
        """Returns the size of chunks.

        Args:
            None

        Returns:
            chunk_shape (tuple [int, int]):
                (Number of channels, time chunk size)
        """
        return (self.ch_range.length(), self.chunk_size)

    def batch_generator(self):
        """Loads the next time-chunk from the data file.

        This implementation works as a generator.

        The ECEI data is usually normalized to a fixed offset, calculated using data
        at the beginning of the stream.

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
            # Use start-time stored in attrs['TriggerTime'][0]
            # Use end-time stored in attrs['TriggerTime'][1]
            tend = min(self.attrs['TriggerTime'][1],
                       self.attrs['TriggerTime'][0] + 5_000_000 / self.attrs['SampleRate'])
            tb_chunk = timebase_streaming(self.attrs['TriggerTime'][0],
                                          tend,
                                          self.attrs['SampleRate'],
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
                self._read_from_hdf5(_chunk_data, current_chunk * self.chunk_size,
                                     (current_chunk + 1) * self.chunk_size)

            current_chunk = ecei_chunk(_chunk_data, tb_chunk, params=self.attrs)
            yield current_chunk


# End of file dataloader.py
