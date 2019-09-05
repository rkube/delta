#-*- coding: UTF-8 -*-

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