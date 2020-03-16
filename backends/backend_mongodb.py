#Coding: UTF-8 -*-

import datetime
import numpy as np
import pickle
import string
import random
import logging
import json
import uuid

import pymongo 
import gridfs
from bson.binary import Binary
from os import mkdir
from os.path import isdir, join

from .backend import backend, serialize_dispatch_seq


class backend_mongodb(backend):
    """
    Author: Ralph Kube

    This defines an access to the mongodb storage backend.
    """
    def __init__(self, cfg_mongo):
        # Connect to mongodb

        self.client = pymongo.MongoClient("mongodb://mongodb07.nersc.gov/delta-fusion", 
                                          username = cfg_mongo["storage"]["username"],
                                          password = cfg_mongo["storage"]["password"])
        db = self.client.get_database()
        self.datadir = join(cfg_mongo["storage"]["datadir"], cfg_mongo["run_id"])

        # Analysis data is either stored in gridFS(slow!) or numpy.
        assert cfg_mongo["datastore"] in ["gridfs", "numpy"]

        if cfg_mongo["datastore"] == "numpy":
            # Initialize storage directory
            if (isdir(self.datadir) == False):
                try:
                    mkdir(self.datadir)
                except:
                    raise ValueError(f"Could not access path {self.datadir}")
            self.fs = None

        elif cfg_mongo["datastore"] == "gridfs":
            # Initialize gridFS
            self.fs = gridfs.GridFS(db)       
        
        try:
            self.collection = db.get_collection("test_analysis_" + cfg_mongo['run_id'])
        except:
            print("Could not get a collection")

            

    def store_metadata(self, cfg, dispatch_seq):
        """Stores the metadata to the database

        Parameters
        ----------
        cfg: The configuration of the analysis run
        dispatch_seq: The serialized task dispatch sequence
        """

        logger = logging.getLogger("DB")
        logger.debug("backend_mongodb: entering store_metadata")

        j_str = serialize_dispatch_seq(dispatch_seq)
        # Put the channel serialization in the corresponding key
        j_str = '{"channel_serialization": ' + j_str + '}'
        j = json.loads(j_str)
        # Adds the channel_serialization key to cfg
        cfg.update(j)
        cfg.update({"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %X UTC")})
    
        cfg.update({"description": "metadata"})

        try:
            result = self.collection.insert_one(cfg)
        except pymongo.errors.PyMongoError as e:
            logger.error("An error has occurred in store_metadata:: ", e)

        return result.inserted_id


    def store_task(self, task, future=None, dummy=True):
        """Stores data from an analysis task in the mongodb backend.

        The data anylsis results from analysis_task object are evaluated in this method.

        Parameters
        ----------
        task: analysis_task object. 
        dummy: bool. If true, do not insert the item into the database
        
        Returns
        -------
        None
        """

        # Gather the results from all futures in the task
        # This locks until all futures are evaluated.
        result = []
        for future in task.futures_list:
            result.append(future.result())
        result = np.array(result)

        # Write results to the backend
        storage_scheme = task.storage_scheme
        # Add a time stamp to the scheme
        storage_scheme["time"] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if dummy:
            storage_scheme["results"] = result
            print(storage_scheme)
        else:
            storage_scheme["results"] = Binary(pickle.dumps(result))
            self.collection.insert_one(storage_scheme)

        return None


    def store_data(self, data, info_dict):
        """Stores data in mongodb

        Parameters
        ----------
        data: ndarray, float.
        info_dict: Dictionary with metadata to store
        cfg: delta configuration object
        """

        
        

        if cfg["datastore"] == "gridfs":
            # Create a binary object and store it in gridfs
            fid = self.fs.put(Binary(pickle.dumps(data)))
            info_dict.update({"result_gridfs": fid})
        
        elif cfg["datastore"] == "numpy":
            # Create a unique file-name
            unq_fname = uuid.uuid1()
            unq_fname = unq_fname.__str__() + ".npz"
            np.savez(join(self.datadir, unq_fname), data=data)

        info_dict.update({"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")})
        info_dict.update({"description": "analysis results"})
        info_dict.update({"unique_filename": unq_fname})

        try:
            inserted_id = self.collection.insert_one(info_dict)

        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise


    def store_one(self, item):
        """Wrapper to store an item"""

        self.collection.insert_one(item)


# End of file mongodb.py