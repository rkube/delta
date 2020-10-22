# -*- Encoding: UTF-8 -*-

import logging
import time
from preprocess.helpers import get_preprocess_routine


class preprocessor():
    """Defines a pre-processing pipeline"""

    def __init__(self, executor, cfg_preprocess):
        """Configures the pre-processing pipeline from file.

        Parameters:
        -----------
        cfg_preprocess: Dictionary
        """

        self.logger = logging.getLogger("simple")
        # Iterate over pre-process routines defined in the configuration
        # For each item, add the appropriate pre-processing routine to the list.

        self.executor = executor
        self.preprocess_list = []
        for key, params in cfg_preprocess.items():
            try:
                pre_task = get_preprocess_routine(key, params)
                self.preprocess_list.append(pre_task)
            except NameError as e:
                self.logger.error(f"Could not find suitable pre-processing routine: {e}")
                continue
            self.logger.info(f"Added {key} to preprocessing")

    def submit(self, data):
        """Launches preprocessing routines on the executor."""

        tic = time.perf_counter()
        for item in self.preprocess_list:
            print("Preprocessing: ", item)
            data = item.process(data, self.executor)

        toc = time.perf_counter()
        tictoc = toc - tic
        self.logger.info(f"Preprocessing took {tictoc:6.4f}s. Returning: {type(data)}")

        return data

# End of file preprocess.py