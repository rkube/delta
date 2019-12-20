# -*- Encoding: UTF-8 -*-

from mpi4py import MPI 
from mpi4py.futures import MPIPoolExecutor
import numpy as np 
import logging
import threading
import queue
import timeit

import attr
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


@attr.s 
class AdiosMessage:
    """ Defines data chunks as read from adios."""
    tstep_idx = attr.ib(repr=True)
    data       = attr.ib(repr=False)


def calc2(i, data):
    logging.info(f"calc2: i = {i}, data = {data}")

    return np.random.rand()


def consume(Q):
    while True:
        msg = Q.get()
        logging.info(f"Consumed message {msg}, {msg.tstep_idx}")
        if msg.tstep_idx == -1:
            Q.task_done()
            break

        with MPIPoolExecutor(max_workers=4) as executor:
            #for res in executor.map(calc, range(5)):
            for res in [executor.submit(calc2, i, msg.data) for i in range(5)]:
                logging.info(f"i = {msg.tstep_idx}, res = {res.result()}")

        Q.task_done()
        logging.info(f"Processed message")


def main():

    dq = queue.Queue()
    msg = AdiosMessage(0, None)
    data = np.zeros([192, 512, 38], dtype=np.complex128)

    worker = threading.Thread(target=consume, args=(dq, ))
    worker.start()

    for i in range(5):
        logging.info(f"Time step {i}")
        msg = AdiosMessage(tstep_idx=i, data=data)
        dq.put(msg)

    dq.put(AdiosMessage(-1, None))

    worker.join()
    dq.join()


if __name__ == "__main__":
    main()



# End of file mpi_processor_mockup.py