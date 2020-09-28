from distributed import Client
import numpy as np 
import logging
import threading
import queue
import timeit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def consume(Q, dask_client):
    while True:
        (i, data) = Q.get()
        if i == -1:
            Q.task_done()
            break

        tic_sc = timeit.default_timer()
        future = dask_client.scatter(data, broadcast=True, direct=True)
        toc_sc = timeit.default_timer()
        logging.info(f"Scatter took {(toc_sc - tic_sc):6.4f}s")

        Q.task_done()


def main():
    dq = queue.Queue()
    msg = None
    data = np.zeros([192, 512, 38], dtype=np.complex128)
    dask_client = Client(scheduler_file="/scratch/gpfs/rkube/dask_work/scheduler.json")
    worker = threading.Thread(target=consume, args=(dq, dask_client))
    worker.start()

    for i in range(5):
        data = data + np.random.uniform(0.0, 1.0, data.shape)
        dq.put((i, data))

    dq.put((-1, None))

    worker.join()
    dq.join()


if __name__ == "__main__":
    main()