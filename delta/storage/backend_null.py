# Encoding: UTF-8 -*-

"""Dummy storage that does nothing."""

import logging
from storage import backend


class backend_null(backend):
    """Dummy backend. Stores no data."""

    def __init__(self, cfg):
        """Initializes."""
        super().__init__()
        self.logger = logging.getLogger("simple")

    def store_one(self, item):
        """Does nothing."""
        pass

    def store_data(self, data, info_dict):
        """Logs the function call and exits."""
        self.logger.info("storage finished:", info_dict)
        return None

    def store_metadata(self, cfg, dispatch_seq):
        """Stores nothing."""
        self.logger.info("store_metadata finished")
        return None

# End of file backend_null.py
