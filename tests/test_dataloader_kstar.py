# -*- Encoding: UTF-8 -*-

"""Verify that dataloaders works correctly."""

import mock
import numpy as np


def load_dummy_data(cls, array, idx_start, idx_end):
    """Replaces _read_from_hdf5 in dataloader."""
    num_channels = array.shape[0]

    dummy_data = np.random.uniform(3.0, 4.0, [num_channels, idx_end - idx_start])
    array[:] = dummy_data[:]


def test_dataloader_ecei_cached(config_all):
    """Verify that _dataloader_ecei works correctly when using cached data."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    from delta.sources.dataloader import _loader_ecei

    cfg_all = config_all
    cfg_all["diagnostic"]["datasource"]["num_chunks"] = 5
    # Instantiate a loader where _read_from_hdf5 is replaced with load_dummy_data
    with mock.patch.object(_loader_ecei, "_read_from_hdf5", new=load_dummy_data):
        my_loader = _loader_ecei(cfg_all, cache=True)

        for batch in my_loader.batch_generator():
            # Mean should be roughly 3.5, depending on what use as dummy data
            assert(np.abs(np.mean(batch.data) - 3.5) < 1e-2)


def test_dataloader_ecei_nocache(config_all):
    """Verify that _dataloader_ecei works correctly when using cached data."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    # # Import packages as delta.... so that we can run pytest as 
    from delta.sources.dataloader import _loader_ecei

    cfg_all = config_all
    # Instantiate a loader where _read_from_hdf5 is replaced with load_dummy_data
    with mock.patch.object(_loader_ecei, "_read_from_hdf5", new=load_dummy_data):
        my_loader = _loader_ecei(cfg_all, cache=False)

        assert(my_loader.get_chunk_shape() == (192, cfg_all["diagnostic"]["datasource"]["chunk_size"]))
        for batch in my_loader.batch_generator():
            # Mean should be roughly 3.5, depending on what use as dummy data
            assert(np.abs(np.mean(batch.data) - 3.5) < 1e-2)


# End of file test_dataloader_kstar.py
