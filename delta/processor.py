# -*- Encoding: UTF-8 -*-

"""This processor implements the one-to-one model using mpi.

To run on an interactive node
srun -n 4 python -m mpi4py.futures processor_mpi_tasklist.py  --config configs/test_all.json

Remember to have adios2 included in $PYTHONPATH

This is the streaming_attrs branch.
"""

from mpi4py import MPI

import logging
import logging.config
import random
import queue
import threading
import time
import string
import json
import yaml
import argparse

from concurrent.futures import ThreadPoolExecutor
from mpi4py.futures import MPIPoolExecutor

from preprocess.preprocess import preprocessor
from analysis.tasks_mpi import task_list
from streaming.reader_mpi import reader_gen
from data_models.helpers import gen_channel_name, gen_var_name, data_model_generator
from storage.backend import get_storage_object


def consume(Q, my_task_list, my_preprocessor):
    """Dispatch work items from the queue on an executor.

    Executed by a local thread.
    """
    logger = logging.getLogger('simple')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logger.info("Starting consume")

    while True:
        try:
            msg = Q.get(timeout=5.0)
        except queue.Empty:
            logger.info("Empty queue after waiting until time-out. Exiting")
            break

        logger.info(f"Rank {rank}: Consumed: {msg.tb} Got data type {type(msg)}")
        msg = my_preprocessor.submit(msg)
        my_task_list.submit(msg)
        Q.task_done()

    logger.info("Task done")


def main():
    """Procesess a stream of data chunks on an executor."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Parse command line arguments and read configuration file
    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis" +
                                     "tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file',
                        default='configs/config_null.json')
    parser.add_argument("--num_threads_preprocess", type=int,
                        help="Number of threads used in preprocessing executor",
                        default=4)
    parser.add_argument("--num_threads_analysis", type=int,
                        help="Number of threads used in analysis executor",
                        default=16)
    parser.add_argument("--num_worker_threads", type=int,
                        help="Number of worker threads that consume incoming data chunks",
                        default=4)
    args = parser.parse_args()

    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()

    # Load logger configuration from file:
    # http://zetcode.com/python/logging/
    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger = logging.getLogger('simple')

    # Create PoolExecutors for preprocessing and analysis
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=60)
    # PoolExecutor for pre-processing, on-node.
    # executor_pre = MPIPoolExecutor(max_workers=4)
    executor_pre = ThreadPoolExecutor(max_workers=args.num_threads_preprocess)
    # PoolExecutor for data analysis. off-node
    executor_anl = MPIPoolExecutor(max_workers=args.num_threads_analysis)

    stream_varname = gen_var_name(cfg)[rank]

    cfg["run_id"] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    cfg["run_id"] = "ABC234"
    cfg["storage"]["run_id"] = cfg["run_id"]
    logger.info(f"Starting run {cfg['run_id']}")

    # Instantiate a storage backend and store the run configuration and task configuration
    store_type = get_storage_object(cfg["storage"])
    store_backend = store_type(cfg["storage"])
    store_backend.store_one({"run_id": cfg['run_id'], "run_config": cfg})

    #TODO: (RMC)  Should this be moved to where cfg updated? (would allow updating channels to process remotely)
    reader = reader_gen(cfg["transport_nersc"], gen_channel_name(cfg["diagnostic"]))

    dq = queue.Queue()
    msg = None

    tic_main = time.perf_counter()

    # # reader.Open() is blocking until it opens the data file or receives the
    # # data stream. Put this right before entering the main loop
    logger.info(f"{rank} Waiting for generator")
    reader.Open()
    stream_attrs = reader.get_attrs("cfg")
    # TODO: Fix this somehow.
    stream_attrs["SampleRate"] = stream_attrs["SampleRate"] * 1e3

    data_model_gen = data_model_generator(cfg["diagnostic"])
    my_preprocessor = preprocessor(executor_pre, cfg)
    my_task_list = task_list(executor_anl, cfg["task_list"], cfg["storage"])

    worker_thread_list = []
    for _ in range(args.num_worker_threads):
        new_worker = threading.Thread(target=consume, args=(dq, my_task_list, my_preprocessor))
        new_worker.start()
        worker_thread_list.append(new_worker)

    logger.info("Starting main loop")
    rx_list = []
    while True:
        stepStatus = reader.BeginStep()
        if stepStatus:
            # Read data
            stream_data = reader.Get(stream_varname, save=False)
            if reader.CurrentStep() in [0, 140]:
                rx_list.append(reader.CurrentStep())

                # Create a datamodel instance from the raw data and push into the queue
                msg = data_model_gen.new_chunk(stream_data, stream_attrs, reader.CurrentStep())
                dq.put_nowait(msg)
                logger.info(f"Published tidx {reader.CurrentStep()}")
            reader.EndStep()
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            break

        if reader.CurrentStep() > 142:
            logger.info("End of the line. Exiting")
            break
    dq.join()
    logger.info("Queue joined")

    logger.info("Exiting main loop")
    for thr in worker_thread_list:
        thr.join()

    logger.info("Workers have joined")

    # Shutdown the executioner
    executor_anl.shutdown(wait=True)
    executor_pre.shutdown(wait=True)
    # executor.shutdown(wait=True)

    toc_main = time.perf_counter()
    logger.info(f"Run {cfg['run_id']} finished in {(toc_main - tic_main):6.4f}s")
    logger.info(f"Processed {len(rx_list)} time_chunks: {rx_list}")


if __name__ == "__main__":
    main()

# End of file processor_mpi.py
