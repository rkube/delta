# Encoding: UTF-8 -*-


"""
Author: Ralph Kube
Tests connection to mongodb
"""

from pymongo import MongoClient
from urllib.parse import quote_plus


client = MongoClient("mongodb://mongodb07.nersc.gov/delta-fusion", 
                     username="delta-fusion_admin",
                     password="eeww33ekekww212aa")

db = client.get_database()
print("Connected to database {0:s}".format(db.name))

collection = db.test_analysis_ABC126

# Print everything in this collection
print("Items in collection test_analysis:")
for item in collection.find():
    print(item)


