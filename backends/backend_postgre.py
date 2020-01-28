#Encoding: UTF-8 -*-


import datetime
import numpy as np 
import logging 

import psycopg2

from .backend import backend, serialize_dispatch_seq


class backend_postgre(backend):
    """Defines the Postgresql backend

    Author: Ralph Kube
    """
    def __init__(self, cfg_postgre):
        # Connect to postgresql server

        logger = logging.getLogger("DB")

        self.conn = psycopg2.connect(host=cfg_postgre["hostname"],
                                     database=cfg_postgre["database"],
                                     user=cfg_postgre["username"],
                                     password=cfg_postgre["password"])

        logger.info(f"Connected to DB {cfg_postgre['database']} on {cfg_postgre['hostname']}")
                    


    def store_metadata(self, cfg, dispatch_seq):
        """Stores analysis metadata

        Parameters
        ----------
        cfg: Configuration of the analysis run
        dispatch_seq: The serialized task dispatch sequence
        """

        logger = logging.getLogger("DB")
        logger.debug("backend_postgre: entering store_metadata")

        cur = self.conn.cursor()
        

