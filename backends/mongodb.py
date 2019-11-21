#Coding: UTF-8 -*-

from pymongo import MongoClient
from bson.binary import Binary
import datetime

# Schemes that describe how results from a given analysis is stored in the database
#schemes_dict={"power_spectrum": lambda task, result: {"routine_name": task.task_name,
#                                                      "channel": task.channel_list[0],
#                                                      "kwargs": task.kw_dict,
#                                                      "f": result[0].tolist(),
#                                                      "Pxx": result[1].tolist()},
#              "cross_phase": lambda task, result: {"analysis_name": task.analysis,
#                                                   "ref_channels" : task.ref_channels.to_str(),
#                                                   "cross_channels" : task.x_channels.to_str(),
#                                                   "result": ["{0:8.6f}".format(r) for r in result]}}


class mongodb_backend():
    def __init__(self, rank=0, channel_list=[]):
        # Connect to mongodb
        self.client = MongoClient("mongodb://mongodb07.nersc.gov/delta-fusion", 
                                  username="delta-fusion_admin",
                                  password="eeww33ekekww212aa")

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
        result_list = []
        for future in task.futures_list:
            result_list.append(future.result())

        # Write results to the backend
        #for future in task.futures_list:
        storage_scheme = task.storage_scheme
        # Add a time stamp to the scheme
        storage_scheme["time"] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        storage_scheme["results"] = ["{0:8.6f}".format(r) for r in result_list]

        if dummy:
            print(storage_scheme)
        else:
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