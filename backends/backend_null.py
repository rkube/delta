# Encoding: UTF-8 -*-

import logging
from .backend import backend

class backend_null(backend):
    """
    Author: Ralph Kube

    backend_null mimicks the other backends but doesn't call any store routine.
    No output is written

    """
    def __init__(self, cfg):
        super().__init__()

    def store(self, chunk_data, chunk_info):
        """Stores nothing"""
        logger = logging.getLogger()
        logger.info("storage finished:", chunk_info)

        return None

    def store_one(self, item):
        pass

    def store_data(self, data, info_dict):
        pass

    def store_metadata(self, cfg, dispatch_seq):
        """Stores nothing"""
        logger = logging.getLogger()

        logger.info("store_metadata finished")

        return None