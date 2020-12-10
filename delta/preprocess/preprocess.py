# -*- Encoding: UTF-8 -*-

"""Implements pre-processing pipeline."""


import logging
import time
from preprocess.helpers import get_preprocess_routine
from storage.backend import get_storage_object
from data_models.kstar_ecei import get_geometry


class preprocessor():
    """Defines a pre-processing pipeline.

    This class defines a pre-processing pipeline that is serially executed on an
    executor.
    """

    def __init__(self, executor, cfg):
        """Configures the pre-processing pipeline from a dictionary.

        For each key-value pairs in `cfg['preprocess']`, a pre-processing callable
        will be configured and appended list of callables.

        Args:
            executor (PEP-3148-style executor):
                Executor on which all pre-processing will be performed
            cfg: (dict)
                Delta configuration

        Returns:
            None
        """
        self.logger = logging.getLogger("simple")
        # Iterate over pre-process routines defined in the configuration
        # For each item, add the appropriate pre-processing routine to the list.

        self.cfg = cfg
        self.executor = executor
        self.preprocess_list = []
        for key, pre_params in cfg["preprocess"].items():
            try:
                pre_task = get_preprocess_routine(key, pre_params)
                self.preprocess_list.append(pre_task)
            except NameError as e:
                self.logger.error(f"Could not find suitable pre-processing routine: {e}")
                continue
            self.logger.info(f"Added {key} to preprocessing")

        self.metadata_stored = False

    def _store_metadata(self, timechunk):
        """Stores metadata that becomes available from timechunks.

        This was added in a hurry in the KSTAR experiments.
        """
        cfg_storage = self.cfg["storage"]
        rpos_arr, zpos_arr, _ = get_geometry(timechunk.params)
        # Note: We need to convert np.bool to bool too when converting timechunk.bad_channels
        # into a list
        chunk_t0, chunk_t1 = timechunk.tb.get_trange()
        chunk_metadata = {"bad_channels": [bool(c) for c in timechunk.bad_channels],
                          "tstart": chunk_t0,
                          "tend": chunk_t1,
                          "dt": timechunk.tb.dt,
                          "rarr": list(rpos_arr),
                          "zarr": list(zpos_arr),
                          "chunk_idx": timechunk.tb.chunk_idx,
                          "description_new": "chunk_metadata"}

        storage_class = get_storage_object(cfg_storage)
        storage = storage_class(cfg_storage)
        storage.store_metadata(chunk_metadata)

    def submit(self, timechunk):
        """Launches preprocessing routines on the executor.

        Args:
            timechunk (timechunk):
                A time-chunk of 2D image data.

        Returns:
            timechunk (timechunk)
                Pre-processed timechunk data
        """
        self.logger.info(f"Start pre-processing of chunk. attrs={timechunk.params}")

        # if not self.metadata_stored:
        self._store_metadata(timechunk)
            # self.metadata_stored = True

        tic = time.perf_counter()
        for item in self.preprocess_list:
            timechunk = item.process(timechunk, self.executor)

        toc = time.perf_counter()
        tictoc = toc - tic
        self.logger.info(f"Preprocessing for chunk {timechunk.tb.chunk_idx:03d} took {tictoc:6.4f}s.\
             Returning: {type(timechunk)}")

        return timechunk

# End of file preprocess.py
