# Encoding: UTF-8 -*-

"""We test how to efficiently submit tasks with large arguments to a dask cluster
https://stackoverflow.com/questions/41471248/how-to-efficiently-submit-tasks-with-large-arguments-in-dask-distributed
"""


import numpy as np
from distributed import Client, progress

dask_client = Client(scheduler_file="/global/cscratch1/sd/rkube/scheduler.json")

data = np.random.uniform(0.0, 1.0, [10_000_000, 16])
data_fut = dask_client.scatter(data, broadcast=True)


def myfun(data, idx):
    print("**myfun: data.shape = ", data.shape, ", idx = ", idx)
    return(data[:, idx].sum())


res = [dask_client.submit(myfun, data_fut, i) for i in range(16)]

print(dask_client.gather(res))

# End of file test_scatter.py