# -*- Encoding: UTF-8 -*-

"""Verify that dataloaders works correctly.

TODO:
1. Implement tests to verify that the correct attributes are read from HDF5.
   Delta now reads this section

  from HDF5 attributes instead of hard-coding it in the config file.


"""

try:
    import mock
except ImportError:
    from unittest import mock

from tests.conftest import stream_attrs_018431
import numpy as np


def read_from_hdf5_dummy(cls, array, idx_start, idx_end):
    """Replaces _read_from_hdf5 in dataloader."""
    num_channels = array.shape[0]

    dummy_data = np.random.uniform(3.0, 4.0, [num_channels, idx_end - idx_start])
    array[:] = dummy_data[:]


def read_attributes_from_hdf5_dummy(cls, attrs):
    """Sets dummy attributes.

    ToDo: We use hard-coded stream_attrs_018431. This should really be the
    fixture defined in conftest. But I don't know how to use the fixture in this
    function and pass the argument in the mock.patch.multiple decorator.
    """
    cls.attrs = attrs


def test_dataloader_ecei_cached(config_all, stream_attrs_018431):
    """Verify that _dataloader_ecei works correctly when using cached data."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    from delta.sources.loader_kstarecei import loader_kstarecei

    # Generate a read_attribute_dummy using the stream_attrs_... fixture.
    # This needs to be a closure, as we can't pass the fixture as a function
    # argument when we patch using the decorator call below:
    def read_attrs_dummy(cls):
        return read_attributes_from_hdf5_dummy(cls, stream_attrs_018431)

    cfg_all = config_all
    cfg_all["diagnostic"]["datasource"]["num_chunks"] = 5
    # Instantiate a loader where _read_from_hdf5 is replaced with load_dummy_data
    # with mock.patch.object(_loader_ecei, "_read_from_hdf5", new=read_from_hdf5_dummy):
    with mock.patch.multiple(loader_kstarecei, _read_from_hdf5=read_from_hdf5_dummy,
                             _read_attributes_from_hdf5=read_attrs_dummy):
        my_loader = loader_kstarecei(cfg_all)

        for batch in my_loader.batch_generator():
            # Mean should be roughly 3.5, depending on what use as dummy data
            assert(np.abs(np.mean(batch.data) - 3.5) < 1e-2)


def test_dataloader_ecei_nocache(config_all, stream_attrs_018431):
    """Verify that _dataloader_ecei works correctly when using cached data."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    # # Import packages as delta.... so that we can run pytest as
    from delta.sources.loader_kstarecei import loader_kstarecei

    # Generate a read_attribute_dummy using the stream_attrs_... fixture.
    # This needs to be a closure, as we can't pass the fixture as a function
    # argument when we patch using the decorator call below.
    def read_attrs_dummy(cls):
        return read_attributes_from_hdf5_dummy(cls, stream_attrs_018431)

    cfg_all = config_all
    # Instantiate a loader where _read_from_hdf5 is replaced with load_dummy_data
    # with mock.patch.object(_loader_ecei, "_read_from_hdf5", new=read_from_hdf5_dummy):
    with mock.patch.multiple(loader_kstarecei, _read_from_hdf5=read_from_hdf5_dummy,
                             _read_attributes_from_hdf5=read_attrs_dummy):

        my_loader = loader_kstarecei(cfg_all)

        assert(my_loader.get_chunk_shape() == (192, cfg_all["diagnostic"]["datasource"]["chunk_size"]))
        for batch in my_loader.batch_generator():
            # Mean should be roughly 3.5, depending on what use as dummy data
            assert(np.abs(np.mean(batch.data) - 3.5) < 1e-2)


# End of file test_dataloader_kstar.py
