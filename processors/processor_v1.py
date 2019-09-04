#-*- coding: UTF-8 -*-

import faust
import numpy as np
import pickle

import sys
sys.path.append("/global/homes/r/rkube/repos/fluctana")
from specs import fftbins

app = faust.App('processor_v1', broker='kafka://localhost:9092', value_serializer='raw', store="memory://")

delta_topic = app.topic('H2202')

@app.agent(delta_topic)
async def consume(data):
    async for obj in data:
        d = pickle.loads(obj)
        print("received data: ", type(d))


# End of file processor_v1.py