#-*- coding: UTF-8 -*-

"""
Below is old information for running the kafka/faust implementation:
Start/Stop zookeper and kafka: https://gist.github.com/piatra/0d6f7ad1435fa7aa790a
#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Please supply topic name"
  exit 1
fi

nohup bin/zookeeper-server-start.sh -daemon config/zookeeper.properties > /dev/null 2>&1 &
sleep 2
nohup bin/kafka-server-start.sh -daemon config/server.properties > /dev/null 2>&1 &
sleep 2

bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic $1
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic parsed
"""

import faust
import numpy as np
import pickle

from pymongo import MongoClient
from bson.binary import Binary
from scipy.signal import spectrogram


channel = 2202

app = faust.App('processor_v1', broker='kafka://localhost:9092', value_serializer='raw', store="memory://")
delta_topic = app.topic('H{0:4d}'.format(channel))

client = MongoClient()
db = client.ECEI
mycol = db.Diagnsotics

t_start = None
t_end = None


@app.agent(delta_topic)
async def consume(data):
    tix = 0
    async for obj in data:
        rec_data = pickle.loads(obj)
        #print("received data: ", type(d))

        res = spectrogram(rec_data)
        print(res[0].shape, res[1].shape, res[2].shape)

        post = {"diagnostic": "ECEI",
                "channel": channel,
                "tstart": t_start,
                "tend": t_end,
                "consumer_id = ": "diag_spectrogram",
                "data": Binary(pickle.dumps(res))}

        mycol.insert_one(post)


        #fig = plt.figure()
        #ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
        #ax.contourf(res[1], res[0], res[2])
        #fig.savefig("fig{0:03d}.png".format(tix))
        tix += 1



# End of file processor_v1.py