#Coding: UTF-8 -*-

#from mpi4py import MPI
from pymongo import MongoClient
from bson.binary import Binary

class mongodb_backend():
    def __init__(self, rank, channel_list):
        self.rank = rank
        self.channel_list = channel_list
        # Connect to mongodb
        client = MongoClient("mongodb07.nersc.gov")


    def store(self, analysis, result):
        """Stores analysis data 

        Input:
        ======
        channel_list: List of channels 
        analysis: dictionary, name and parameters for called analysis routine
        result: Result of the analysis routine
        

        Returns:
        ========
        None
        """

        print("Storing data")

        return None


# End of file mongodb.py