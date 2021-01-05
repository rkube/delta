# -*- Encoding: UTF-8 -*-

"""Helper functions common to all data models."""

import logging
import numpy as np
from itertools import filterfalse

import more_itertools

from data_models.kstar_ecei import ecei_chunk
from data_models.channels_2d import channel_2d, channel_range, channel_pair
from data_models.timebase import timebase_streaming


class data_model_generator():
    """Callable that wraps a block of data into a data-model object."""

    def __init__(self, cfg_diagnostic: dict):
        """Sets up data model generation.

        Args:
            cfg_diagnostic: dict,
                Diagnostic section of the config file

        Raises:
            ValueError:
                Field 'name' specified in cfg_diagnostic could not be matched
                to an existing data_model

        Used keys from cfg_all:
            * diagnostic.datasource.chunk_size
            * diagnostic.datasource.t_norm
            * diagnostic.name
        """
        self.logger = logging.getLogger("simple")
        self.cfg = cfg_diagnostic

        # Generate start/stop time for timebase
        self.chunk_size = cfg_diagnostic["datasource"]["chunk_size"]
        # Callable that performs normalization. This is instantiated once data from
        # the time interval t_norm is read
        self.normalize = None
        self.t_norm = self.cfg["datasource"]["t_norm"]

        if self.cfg["name"] == "kstarecei":
            self.data_type = ecei_chunk
        elif self.cfg["name"] == "nstxgpi":
            self.data_type = None
        else:
            raise ValueError(f"No data model for diagnostic {cfg_diagnostic['name']}")

    def new_chunk(self, stream_data: np.array, stream_attrs: dict, chunk_idx: int):
        """Generates a data model from new chunk of streamed data.

        Args:
            stream_data (np.array):
                New data chunk read from :class: reader_gen.
            stream_attr (dict):
                Stream attributes, taken from :class: reader_gen.
            chunk_idx (int):
                Sequence index of this time chunk in the stream

        Returns:
            None
        """
        # Generate a time-base and a data model
        if self.cfg["name"] == "kstarecei":
            # Adapt configuration file parameters for use in timebase_streaming constructor
            self.logger.info(f"New chunk: attrs = {stream_attrs}, chunk_idx = {chunk_idx}")
            tb_chunk = timebase_streaming(stream_attrs["TriggerTime"][0],
                                          stream_attrs["TriggerTime"][1],
                                          stream_attrs["SampleRate"],
                                          self.chunk_size, chunk_idx)
            chunk = self.data_type(stream_data, tb_chunk, stream_attrs)

            # Determine whether we need to normalize the data
            tidx_norm = [tb_chunk.time_to_idx(t) for t in self.t_norm]

            if (tidx_norm[0] is not None) and (tidx_norm[1] is not None) and self.normalize is None:
                # TODO: Here we create a normalization object using explicit values for the
                # normalization It may be better to just pass the data and let the normalization
                # object figure out how to calculate the needed constants. This would be the best
                # way to allow different normalization.
                data_norm = stream_data[:, tidx_norm[0]:tidx_norm[1]]
                self.normalize = normalize_mean(data_norm)
                self.logger.info(f"Calculated normalization using\
                                 {tidx_norm[1] - tidx_norm[0]} samples.")

            elif self.normalize is not None:
                self.normalize(chunk)
            else:
                self.logger.info(f"new_chunk: {chunk_idx}: self.normalize has not been initialized")

            return chunk

        elif self.cfg["name"] == "nstxgpi":
            raise NotImplementedError("NSTX chunk generation not implemented")

        else:
            raise NameError(f"Data model name not understood: {self.cfg['diagnostic']['name']}")


def gen_channel_name(cfg_diagnostic: dict) -> str:
    """Generates a name to be used as the ADIOS channel from diagnostic configuration.

    Args:
        cfg_diagnostic (dict):
            "diagnostic" section from the global Delta configuration

    Returns:
        channel_name (str):
            String to be used as the ADIOS channel name

    Raises:
        ValueError:
            In case the field `name` could not be matched
    """
    if cfg_diagnostic["name"] == "kstarecei":
        experiment = "KSTAR"
        diagnostic = "ECEI"
        shotnr = int(cfg_diagnostic["shotnr"])
        channel_rg = cfg_diagnostic["datasource"]["channel_range"][0]

        channel_name = f"{experiment}_{shotnr:05d}_{diagnostic}_{channel_rg}"
        return channel_name

    else:
        raise ValueError


def gen_channel_range(diag_name: str, chrg: list) -> channel_range:
    """Generates channel ranges for the diagnostics.

    Generates a channel_range object for the diagnostic at hand.

    Args:
        diag_name: str
            Name of the diagnostic
        chrg (tuple[int, int, int, int]):
            ch1_start, ch1_end, ch2_start, ch2_end.

    Returns:
        channel_range (channel_range):
            channel_range configured with appropriate bounds and order for either
            KSTAR ecei or NSTX GPI
    """
    if diag_name == "kstarecei":
        ch1 = channel_2d(chrg[0], chrg[1], 24, 8, order='horizontal')
        ch2 = channel_2d(chrg[2], chrg[3], 24, 8, order='horizontal')
        return channel_range(ch1, ch2)

    elif diag_name == "nstxgpi":
        return None

    else:
        raise ValueError


def gen_var_name(cfg: dict) -> str:
    """Generates a variable name from the diagnostic configuration."""
    # TODO: We should put the device name back in here
    if cfg["diagnostic"]["name"] == "kstarecei":
        return cfg["diagnostic"]["datasource"]["channel_range"]

    else:
        raise ValueError


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes

    This should be legacy code...

    Args:
        iterable (iterable):
            No idea

    Returns:
        element (thingy):
            No idea
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


class normalize_mean():
    """Performs normalization.

    TODO: Write a proper normalizer.
    """

    def __init__(self, data_norm, axis=-1):
        """Stores offset and standard deviation of normalization time series.

        Args:
            data_norm (ndarray):
                Data from which we calculate normalization constants
            axis (int):
                Time-axis.

        Returns:
            None
        """
        self.logger = logging.getLogger("simple")
        self.offlev = np.median(data_norm, axis=-1, keepdims=True)
        self.offstd = data_norm.std(axis=-1, keepdims=True)
        self.siglev = None
        self.sigstd = None
        # self.logger.info(f"Calculating normalization using {data_norm.shape[-1]} samples")
        # self.logger.info(f"Calculated offlev: {self.offlev}")
        # self.logger.info(f"Calculated offstd: {self.offstd}")
        # np.savez("data_norm", data_norm=data_norm, offlev=self.offlev, offstd=self.offstd)

    def __call__(self, chunk):
        """Normalizes data in-place.

        TODO: Write me.

        Args:
          chunk (2d_data):
             Data that will be normalized to siglev and sigstd

        Returns:
            None
        """
        # For these asserts to hold we need to calculate offlev,offstd with keepdims=True

        assert(self.offlev.shape[0] == chunk.shape[chunk.axis_ch])
        assert(self.offstd.shape[0] == chunk.shape[chunk.axis_ch])
        assert(self.offlev.ndim == chunk.data.ndim)
        assert(self.offstd.ndim == chunk.data.ndim)
        self.logger.info("Normalizing current_chunk")

        # Attach offlev and offstd to the chunk
        chunk.offlev = self.offlev
        chunk.offstd = self.offstd

        chunk.data[:] = chunk.data - self.offlev
        chunk.siglev = np.median(chunk.data, axis=chunk.axis_t, keepdims=True)
        chunk.sigstd = chunk.data.std(axis=chunk.axis_t, keepdims=True)
        chunk.data[:] = chunk.data / chunk.data.mean(axis=chunk.axis_t, keepdims=True) - 1.0
        chunk.is_normalized = True
        chunk.mark_bad_channels(verbose=True)

        return None


def get_dispatch_sequence(ref_channels, cmp_channels, niter=128):
    """Returns an a list of iterables that span all unique combinations of ref_ch x cmp_ch.

    Args:
        ref_channels [int, int, int int]:
            List that describes the reference channels. start_h, start_v, end_h, end_v

        cmp_channels [int, int, int, int]:
            List that describes the compare channels. start_h, start_v, end_h, end_v
        niter (int):
            Length of the sub-lists we split the list of channel pairs into.

    Returns:
        all_chunks (???):
            Chunked dispatch sequence.
    """
    # TODO: remove hard-coded kstarecei string.
    ref_channel_rg = gen_channel_range("kstarecei", ref_channels)
    cmp_channel_rg = gen_channel_range("kstarecei", cmp_channels)

    # Construct a list of unique channels
    # F.ex. we have ref_channels [(1,1), (1,2), (1,3)] and cmp_channels = [(1,1), (1,2)]
    # The unique list of channels is then
    # (1,1) x (1,1), (1,1) x (1,2)
    # (1,2) x (1,2) !!! Omits (1,2) x (1,1)
    # (1,3) x (1,1)
    # (1,3) x (1,2)
    channel_pairs = [channel_pair(cr, cx) for cr in ref_channel_rg for cx in cmp_channel_rg]
    # Make a list, so that we don't exhaust the iterator after the first call.
    # self.unique_channels = [i[0] for i in
    #                         more_itertools.distinct_combinations(channel_pairs, 1)]
    unique_channels = [ch for ch in unique_everseen(channel_pairs)]

    all_chunks = list(more_itertools.chunked(unique_channels, niter))
    return(all_chunks)

# End of file helpers.py
