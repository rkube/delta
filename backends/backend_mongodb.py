#Coding: UTF-8 -*-

import datetime
import numpy as np
import pickle

from pymongo import MongoClient
from bson.binary import Binary

from .backend import backend


class backend_mongodb(backend):
    """
    Author: Ralph Kube

    This defines an access to the mongodb storage backend.
    """
    def __init__(self, cfg_mongo):
        # Connect to mongodb
        self.client = MongoClient("mongodb://mongodb07.nersc.gov/delta-fusion", 
                                  username = cfg_mongo["username"],
                                  password = cfg_mongo["password"])
                                  #username="delta-fusion_admin",
                                  #password="eeww33ekekww212aa")

        self.db = self.client.get_database()
        self.collection = self.db.test_analysis
        print("***mongodb_backend*** Connected to database {0:s}".format(self.db.name))
        

    def store(self, task, future=None, dummy=True):
        """Stores data from an analysis task in the mongodb backend.

        The data anylsis results from analysis_task object are evaluated in this method.

        Input:
        ======
        task: analysis_task object. 
        dummy: bool. If true, do not insert the item into the database
        
        Returns:
        ========
        None
        """

        # Gather the results from all futures in the task
        # This locks until all futures are evaluated.
        result = []
        for future in task.futures_list:
            result.append(future.result())
        result = np.array(result)
        print("***Backend.store: result = ", result.shape)

        # Get the 


        # Write results to the backend
        storage_scheme = task.storage_scheme
        # Add a time stamp to the scheme
        storage_scheme["time"] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        for it in task.get_dispatch_sequence():
            print(it)

        

        if dummy:
            storage_scheme["results"] = result
            print(storage_scheme)
        else:
            storage_scheme["results"] = Binary(pickle.dumps(result))
            self.collection.insert_one(storage_scheme)
            print("***mongodb_backend*** Storing...")

        return None


    def store_single(self, task, future, dummy=True):
        """Similar to store, but the future is separate from the task object"""

        #storage_scheme = schemes_dict[task.task_name](task, future.result())
        storage_scheme = task.storage_scheme
        storage_scheme["time"] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        storage_scheme["results"] = ["{0:8.6f}".format(r) for r in result]

        if dummy:
            print(storage_scheme)
        else:
            self.collection.insert_one(storage_scheme)
            print("***mongodb_backend*** Storing...")

# End of file mongodb.py