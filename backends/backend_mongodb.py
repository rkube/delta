#Coding: UTF-8 -*-

import datetime
import numpy as np
import pickle
import string
import random
import logging
import json
import uuid
import time
import traceback

import pymongo 
import gridfs
from bson.binary import Binary
from os import mkdir
from os.path import isdir, join

from .backend import backend, serialize_dispatch_seq

class mongo_connection():
    """Abstraction for mongo_connection using context manager"""

    def __init__(self, cfg_mongo):
        # Parse connection info

        self.conn_info = {}
        with open("mongo_secret", "r") as secret:
            lines = secret.readlines()
            self.conn_info["username"] = lines[0].strip()
            self.conn_info["password"] = lines[1].strip()
            self.conn_info["conn_str"] = lines[2].strip()

        # Parse location for binary data storage
        assert cfg_mongo["datastore"] in ["gridfs", "numpy"]
        self.datastore = cfg_mongo["datastore"]

        # Get name of the run
        assert(len(cfg_mongo["run_id"]) == 6)
        self.coll_str = "test_analysis_" + cfg_mongo["run_id"]

    def __enter__(self):
        """Instantiate a new MongoClient and return the collection"""
        self.client = pymongo.MongoClient(self.conn_info["conn_str"], username=self.conn_info["username"],
                                          password=self.conn_info["password"])

        db = self.client.get_database()
        # if self.datastore == "numpy":
        #     self.datadir = join(cfg_mongo["datadir"], cfg_mongo["run_id"])
        #     # Initialize storage directory
        #     if (isdir(self.datadir) == False):
        #         try:
        #             mkdir(self.datadir)
        #         except:
        #             self.logger.error(f"Could not access path {self.datadir}")
        #             raise ValueError(f"Could not access path {self.datadir}")
        #     self.fs = None

        # elif cfg_mongo["datastore"] == "gridfs":
        #     self.datastore = "gridfs"
        #     # Initialize gridFS
        #     self.fs = gridfs.GridFS(db)     
        try:
            collection = db.get_collection(self.coll_str)
        except:
            self.logger.error(f"Could not access collection {self.coll_str}")

        return (self.client, collection)

    def __exit__(self, exc_type, exc_value, tb):
        """Close connection to MongoDB"""
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            # return False # uncomment to pass exception through

            return True 

        self.db = None
        self.client.close()

        return True



class mongo_storage_numpy():
    """Abstraction for numpy storage using context manager used by backend_mongodb"""
    def __init__(self, cfg_mongo):
        self.datadir = join(cfg_mongo["datadir"], cfg_mongo["run_id"])
        if (isdir(self.datadir) == False):
            try:
                mkdir(self.datadir)
            except:
                self.logger.error(f"Could not access path {self.datadir}")
                raise ValueError(f"Could not access path {self.datadir}")

    def __enter__(self):
        fname = join(self.datadir, uuid.uuid1().__str__() + ".npz")
        return fname

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return True

        return True


class mongo_storage_gridfs():
    """Abstraction for gridfs storage using context manager used by backend_mongodb"""
    def __init__(self, db):
        self.db = db


    def __enter__(self):
        fs = gridfs.GridFS(self.db)
        return fs


    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return True

        return True


class backend_mongodb(backend):
    """
    Author: Ralph Kube

    Defines the MongoDB storage backend. Note that PyMongo is not fork-safe.
    A new MongoClient needs to be instantiated each time store_data is executed on a PoolExecutor
    See
    https://api.mongodb.com/python/current/faq.html#id21

    __init__ is an adaptor pattern that parses the config file and mongo_secret.
    connect() is to be used internally and returns the connection to the database.
    """
    def __init__(self, cfg_mongo):
        """Connect to MongoDB and, if necessary, initializes gridFS."""
        # Connect to mongodb
        self.logger = logging.getLogger("DB")
        self.cfg_mongo = cfg_mongo

        # Parse connection info
        # self.conn_info = {}
        # with open("mongo_secret", "r") as secret:
        #     lines = secret.readlines()
        #     self.conn_info["username"] = lines[0].strip()
        #     self.conn_info["password"] = lines[1].strip()
        #     self.conn_info["conn_str"] = lines[2].strip()

        # Parse location for binary data storage
        assert cfg_mongo["datastore"] in ["gridfs", "numpy"]
        self.datastore = cfg_mongo["datastore"]

        # Get name of the run
        # assert(len(cfg_mongo["run_id"]) == 6)
        # self.coll_str = "test_analysis_" + cfg_mongo["run_id"]
        
        # self.client = pymongo.MongoClient(connection_str, 
        #                                   username=username,
        #                                   password=password)

        # db = self.client.get_database()
        
        # # Analysis data is either stored in gridFS(slow!) or numpy.
        
        # self.datadir = None
        # self.datastore = None
        # if cfg_mongo["datastore"] == "numpy":
        #     self.datastore = "numpy"
        #     self.datadir = join(cfg_mongo["datadir"], cfg_mongo["run_id"])
        #     # Initialize storage directory
        #     if (isdir(self.datadir) == False):
        #         try:
        #             mkdir(self.datadir)
        #         except:
        #             self.logger.error(f"Could not access path {self.datadir}")
        #             raise ValueError(f"Could not access path {self.datadir}")
        #     self.fs = None

        # elif cfg_mongo["datastore"] == "gridfs":
        #     self.datastore = "gridfs"
        #     # Initialize gridFS
        #     self.fs = gridfs.GridFS(db)     
        
        # coll_str = "test_analysis_" + cfg_mongo["run_id"]
        # try:
        #     self.collection = db.get_collection(coll_str)
        # except:
        #     self.logger.error(f"Could not access collection {coll_str}")

    def store_metadata(self, cfg, dispatch_seq):
        """Stores metadata that allows to identify channel pairs with the stored data.

        Parameters
        ----------
        cfg: The configuration of the analysis run
        dispatch_seq: The serialized task dispatch sequence
        """

        j_str = serialize_dispatch_seq(dispatch_seq)
        # Put the channel serialization in the corresponding key
        j_str = '{"channel_serialization": ' + j_str + '}'
        j = json.loads(j_str)
        # Adds the channel_serialization key to cfg
        cfg.update(j)
        cfg.update({"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %X UTC")})
        cfg.update({"description": "metadata"})

        with mongo_connection(self.cfg_mongo) as mongo:
            client, coll = mongo
            try:
                result = coll.insert_one(cfg)
            except pymongo.errors.PyMongoError as e:
               self.logger.error(f"An error has occurred in store_metadata:: {e}")

        return result.inserted_id


    def store_data(self, data, info_dict):
        """Stores arbitrary data in mongodb

        Parameters
        ----------
        data: ndarray, float.
        info_dict: Dictionary with metadata to store
        cfg: delta configuration object
        """


        with mongo_connection(self.cfg_mongo) as mongo:
            client, coll = mongo
            if self.datastore == "gridfs":
                with mongo_storage_gridfs(client.get_database()) as fs:
                    #tic_io = time.perf_counter()
                    fid = fs.put(Binary(pickle.dumps(data)))
                    #toc_io = time.perf_counter()
                    info_dict.update({"result_gridfs": fid})

            elif self.datastore == "numpy":
                with mongo_storage_numpy(self.cfg_mongo) as fname:
                    np.savez(fname, data=data)
                    info_dict.update({"unique_filename": fname})

            info_dict.update({"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")})
            info_dict.update({"description": "analysis results"})

            try:
                inserted_id = coll.insert_one(info_dict)
            except:
                self.logger.error("Unexpected error:", sys.exc_info()[0])


            # if self.datastore == "gridfs":
            #     # Create a binary object and store it in gridfs
            #     fid = self.fs.put(Binary(pickle.dumps(data)))
            #     #tic_io = time.perf_counter()
            #     info_dict.update({"result_gridfs": fid})
            #     #toc_io = time.perf_counter()
            
            # elif self.datastore == "numpy":
            #     # Create a unique file-name
            #     unq_fname = uuid.uuid1()
            #     unq_fname = unq_fname.__str__() + ".npz"
            #     #tic_io = time.perf_counter()
            #     np.savez(join(self.datadir, unq_fname), data=data)
            #     #toc_io = time.perf_counter()
            #     info_dict.update({"unique_filename": unq_fname})

            # #dt_io = toc_io - tic_io
            # except:
            #     logger.error("Unexpected error:", sys.exc_info()[0])

            return None

    def store_one(self, item):
        """Wrapper to store an item"""
        with mongo_connection(self.cfg_mongo) as mongo:
            client, coll = mongo
            coll.insert_one(item)

        return None


# End of file mongodb.py