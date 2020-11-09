# File backends/__init__.py
# -*- Encoding: UTF-8 -*-

"""
Author: Ralph Kube

This folder contains definitions of storage backends.

A parent class is defined in backend.py. The method to store data is called store.
Currently, there is a numpy backend and a mongodb backend.

All actual storage implementations are a derived class of the backend class.
This is to encourage the use of a common interface when storing actual data.
Here is an example:

Let's say we are consuming a message from the queue. see processor_mpi, function consume:


In the main function a backend is defined like this:

>>> store_backend = backend_numpy("/home/rkube/repos/delta/test_data")

This backend is passed to the consume function:
>>> def consume(Q, store_backend, my_fft, task_list, cfg):

When iterating over the tasks to be done for a time-chunk, the store backend is
passed to the task at hand like this:

>>> for task in task_list:
>>>     logging.info("Executing task")
>>>     task.calculate(executor, fft_data)
>>>     task.store_data(store_backend, {"tstep": msg.tstep_idx})

The task can then store its current data together with some meta-data like this:

    backend.store(self.description, res, metadata)

The down-side of this approach is that it is
1) blocking
2) terribly convoluted

"""

# from .backend import backend
# from .backend_numpy import backend_numpy
# from .backend_mongodb import backend_mongodb
# from .backend_null import backend_null

# End of file backends/__init__.py
