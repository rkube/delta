# -*- Encoding: UTF-8 -*-

"""Helper functions common to all data models."""


import numpy as np
from itertools import filterfalse

from data_models.kstar_ecei import ecei_chunk
from data_models.channels_2d import channel_2d, channel_range
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


        """
        self.cfg = cfg_diagnostic

        if self.cfg["name"] == "kstarecei":
            self.data_type = ecei_chunk
        elif self["name"] == "nstxgpi":
            self.data_type = None
        else:
            raise ValueError(f"No data model for diagnostic {cfg_diagnostic['name']}")

    def new_chunk(self, stream_data: np.array, chunk_idx: int):
        """Generates a data model from new chunk of streamed data.

        Args:
            stream_data (np.array):
                New data chunk read from :class: reader_gen.
            chunk_idx (int):
                Sequence index in the stream


        Returns:
            None
        """
        # Generate a time-base and a data model
        if self.cfg["name"] == "kstarecei":
            # Adapt configuration file parameters for use in timebase_streaming
            # constructor
            t_start, t_end, _ = self.cfg["parameters"]["TriggerTime"]
            f_sample = 1e3 * self.cfg["parameters"]["SampleRate"]
            samples_per_chunk = self.cfg["datasource"]["chunk_size"]

            tb = timebase_streaming(t_start, t_end, f_sample, samples_per_chunk, chunk_idx)

            return ecei_chunk(stream_data, tb)

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


def gen_channel_range(cfg_diagnostic: dict, chrg: list) -> channel_range:
    """Generates channel ranges for the diagnostics.

    Generates a channel_range object for the diagnostic at hand.

    Args:
        cfg_diagnostic (dict):
            "diagnostic" section from the global Delta configuration
        chrg (tuple[int, int, int, int]):
            ch1_start, ch1_end, ch2_start, ch2_end.

    Returns:
        channel_range (channel_range):
            channel_range configured with appropriate bounds and order for either
            KSTAR ecei or NSTX GPI
    """
    if cfg_diagnostic["name"] == "kstarecei":
        ch1 = channel_2d(chrg[0], chrg[1], 24, 8, order='horizontal')
        ch2 = channel_2d(chrg[2], chrg[3], 24, 8, order='horizontal')
        return channel_range(ch1, ch2)

    elif cfg_diagnostic["name"] == "nstxgpi":
        return None

    else:
        raise ValueError


def gen_var_name(cfg: dict) -> str:
    """Generates a variable name from the diagnostic configuration."""
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

    def __init__(self, offlev, offstd):
        """Stores offset and standard deviation of normalization time series.

        Args:
            offlev (ndarray):
                channel-wise offset level
            offstd (ndarray):
                channel-wise offset standard deviation

        Returns:
            None
        """
        self.offlev = offlev
        self.offstd = offstd

        self.siglev = None
        self.sigstd = None

    def __call__(self, data):
        """Normalizes data in-place.

        TODO: Write me.

        Args:
          data (2d_data):
             Data that will be normalized to siglev and sigstd

        Returns:
            None
        """
        # For these asserts to hold we need to calculate offlev,offstd with keepdims=True

        assert(self.offlev.shape[:-1] == data.shape[data.axis_t])
        assert(self.offstd.shape[:-1] == data.shape[data.axis_t])
        assert(self.offlev.ndim == data.ndim)
        assert(self.offstd.ndim == data.ndim)

        data[:] = data - self.offlev
        self.siglev = np.median(data, axis=data.axis_t, keepdims=True)
        self.sigstd = data.std(axis=data.axis_t, keepdims=True)

        data[:] = data / data.mean(axis=data.axis_t, keepdims=True) - 1.0

        return None

# End of file helpers.py
