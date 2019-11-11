from pymongo import MongoClient
from urllib.parse import quote_plus
import datetime

client = MongoClient("mongodb://mongodb07.nersc.gov/delta-fusion", 
                     username="delta-fusion_admin",
                     password="eeww33ekekww212aa")

db = client.get_database()
print("Connected to database {0:s}".format(db.name))

collection = db.test_analysis

# Store some test key value pair in test_analysis
timestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
key = "test_insert"


res = collection.insert_one({key: timestr})

if (res.acknowledged):
    print("Insert acknowledged")

print("Items in test_analyis")
for item in collection.find():
    print(item)
