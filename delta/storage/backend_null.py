# Encoding: UTF-8 -*-

"""Dummy storage that does nothing."""

import logging


class backend_null():
    """Dummy backend. Stores no data."""

    def __init__(self, cfg):
        """Initializes."""
        self.logger = logging.getLogger("simple")

    def store_data(self, data, info_dict):
        """Logs the function call and exits."""
        self.logger.debug("storage finished:", info_dict)
        return None

    def store_metadata(self, cfg, dispatch_seq):
        """Stores nothing."""
        self.logger.debug("store_metadata finished")
        return None

    def store_one(self, item):
        """Does nothing."""
        self.logger.debug("store_one called:", item)
        return None

# End of file backend_null.py
