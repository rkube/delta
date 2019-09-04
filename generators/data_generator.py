#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

"""
This is a dummy data generator to send data through kafka into the channel 'ECEI_data'
"""

from kafka import KafkaProducer
import numpy as np
import pickle


producer = KafkaProducer()#bootstrap_servers='localhost:1234')
for _ in range(100):
    #data = np.random.uniform(0.0, 1.0, 23)
    data = np.arange(10, 17, 0.0001)
    # I'm totally not sure how this will play out. Let's go with it and see :)
    producer.send('ECEI_data', pickle.dumps(data))


# End of file data_generator.py