# -*- Encoding: UTF-8 -*-

"""Implements pre-processing pipeline."""


import logging
import time
from preprocess.helpers import get_preprocess_routine


class preprocessor():
    """Defines a pre-processing pipeline.

    This class defines a pre-processing pipeline that is serially executed on the
    processor.
    """

    def __init__(self, executor, cfg):
        """Configures the pre-processing pipeline from a dictionary.

        For each key-value pairs in `cfg_preprocess`, a pre-processing callable
        will be configured and appended list of callables.

        Args:
            executor (PEP-3148-style executor):
                Executor on which all pre-processing will be performed
            cfg_preprocess
                Delta configuration

        Returns:
            None
        """
        self.logger = logging.getLogger("simple")
        # Iterate over pre-process routines defined in the configuration
        # For each item, add the appropriate pre-processing routine to the list.

        self.executor = executor
        self.preprocess_list = []
        for key, params in cfg["preprocess"].items():
            try:
                pre_task = get_preprocess_routine(key, params, cfg["diagnostic"])
                self.preprocess_list.append(pre_task)
            except NameError as e:
                self.logger.error(f"Could not find suitable pre-processing routine: {e}")
                continue
            self.logger.info(f"Added {key} to preprocessing")

    def submit(self, timechunk):
        """Launches preprocessing routines on the executor.

        Args:
            timechunk (timechunk):
                A time-chunk of 2D image data.

        Returns:
            timechunk (timechunk)
                Pre-processed timechunk data
        """
        tic = time.perf_counter()
        for item in self.preprocess_list:
            timechunk = item.process(timechunk, self.executor)

        toc = time.perf_counter()
        tictoc = toc - tic
        self.logger.info(f"Preprocessing for chunk {timechunk.tb.chunk_idx:03d} took {tictoc:6.4f}s.\
             Returning: {type(timechunk)}")

        return timechunk

# End of file preprocess.py
