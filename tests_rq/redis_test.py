#!/use/bin/env python

import sys
sys.path.append("/global/homes/r/rkube/repos/delta")
import requests
from rq import Queue
from redis import Redis   
import numpy as np

import time



test_str = """this is a word counter look at me i can count words"""

# Tell RQ what Redis connection to use
redis_conn = Redis(host="cori02")
q = Queue(connection=redis_conn,
          default_timeout=10)  # no args implies the default queue

# Delay execution of count_words_at_url('http://nvie.com')
job = q.enqueue(np.linalg.norm, np.array([1.0, 2.0]))
time.sleep(0.1)
print(job.result)   # => None

# Now, wait a while, until the worker is finished
time.sleep(5)
print(job.result)   # => 889



# End of file redis_test.py