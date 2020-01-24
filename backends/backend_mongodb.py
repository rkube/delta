#Coding: UTF-8 -*-

import datetime
import numpy as np
import pickle
import string
import random
import logging


import pymongo 
from bson.binary import Binary

from .backend import backend, serialize_dispatch_seq


class backend_mongodb(backend):
    """
    Author: Ralph Kube

    This defines an access to the mongodb storage backend.
    """
    def __init__(self, cfg_mongo):
        # Connect to mongodb
        logging.debug("***mongodb_backend: Initializing mongodb backend")
        self.client = pymongo.MongoClient("mongodb://mongodb07.nersc.gov/delta-fusion", 
                                          username = cfg_mongo["username"],
                                          password = cfg_mongo["password"])
        #                           #username="delta-fusion_admin",
        #                           #password="eeww33ekekww212aa")
        logging.debug("***mongodb_backend: Connection established")
        id_length = 6
        # Generate a unique identifier for this test_run
        self.session_id = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(id_length))
        logging.debug("*** Session_id: ", self.session_id)


        self.client.close()

    #def __del__():
    #    logging.debug("***mongodb_backend: __del__()")
    #    self.client.close()


    def store_metadata(self, cfg, dispatch_seq):
        """Stores the metadata to the database

        Parameters
        ----------
        cfg: The configuration of the analysis run
        dispatch_seq: The serialized task dispatch sequence
        """

        logging.debug("****mongodb_backend: entering store_metadata")

        # db = self.client.get_database()
        # collection = db.get_collection("test_analysis_" + self.session_id)
        # logging.info("***mongodb_backend Connected to database {db.name}")
        # logging.info(f"***mongodb_backend Using collectoin {collection.name}")


        # j_str = serialize_dispatch_seq()
        # # Put the channel serialization in the corresponding key
        # j_str = '{"channel_serialization": ' + j_str + '}'
        # j = json.loads(j_str)
        # # Adds the channel_serialization key to cfg
        # cfg.update(j)
        # cfg.update({"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %X UTC")})
       
        # try:
        #     result = self.collection.insert_one(cfg)
        #     return result.matched_count > 0
        # except pymongo.errors.PyMongoError as e:
        #     logging.info("An error occured when attempting to write metadata: ", e)
        #     return False
        



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

        pass

        # # Gather the results from all futures in the task
        # # This locks until all futures are evaluated.
        # result = []
        # for future in task.futures_list:
        #     result.append(future.result())
        # result = np.array(result)
        # print("***Backend.store: result = ", result.shape)

        # # Get the 


        # # Write results to the backend
        # storage_scheme = task.storage_scheme
        # # Add a time stamp to the scheme
        # storage_scheme["time"] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        # for it in task.get_dispatch_sequence():
        #     print(it)

        

        # if dummy:
        #     storage_scheme["results"] = result
        #     print(storage_scheme)
        # else:
        #     storage_scheme["results"] = Binary(pickle.dumps(result))
        #     self.collection.insert_one(storage_scheme)
        #     print("***mongodb_backend*** Storing...")

        # return None



# End of file mongodb.py