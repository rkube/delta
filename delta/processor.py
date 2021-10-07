# -*- Encoding: UTF-8 -*-

"""This processor implements the one-to-one model using mpi.

To run on an interactive node
srun -n 4 python -m mpi4py.futures processor_mpi_tasklist.py --config configs/test_all.json

Remember to have adios2 included in $PYTHONPATH

This is the streaming_attrs branch.
"""

from mpi4py import MPI

import logging
import logging.config
import time
import queue
import threading
import json
import yaml
import argparse

from concurrent.futures import ThreadPoolExecutor
from mpi4py.futures import MPIPoolExecutor

from preprocess.preprocess import preprocessor
from analysis.task_list import tasklist
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
            msg = Q.get(timeout=120.0)
        except queue.Empty:
            logger.info("Empty queue after waiting until time-out. Exiting")
            break

        logger.info(f"Rank {rank}: Consumed: {msg.tb} Got data type {type(msg)}")
        msg = my_preprocessor.submit(msg)
        my_task_list.execute(msg)
        Q.task_done()

    logger.info("Task done")


def main():
    """Procesess a stream of data chunks on an executor."""
    #comm = MPI.COMM_WORLD

    # Parse command line arguments and read configuration file
    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis" +
                                     "tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file',
                        default='configs/config_null.json')
    parser.add_argument("--num_ranks_preprocess", type=int,
                        help="Number of processes used in preprocessing executor",
                        default=4)
    parser.add_argument("--num_ranks_analysis", type=int,
                        help="Number of processes used in analysis executor",
                        default=4)
    parser.add_argument("--num_queue_threads", type=int,
                        help="Number of worker threads that consume item from the queue",
                        default=4)
    parser.add_argument("--transport", type=str,
                        help="Specifies the transport section used to configure the reader",
                        default="transport_rx")
    parser.add_argument("--run_id", type=str,
                        help="Name of database collection to store analysis results in",
                        required=True)

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

    # PoolExecutor for pre-processing, on-node.
    executor_pre = ThreadPoolExecutor(max_workers=args.num_ranks_preprocess)
    # PoolExecutor for data analysis. off-node
    executor_anl = MPIPoolExecutor(max_workers=args.num_ranks_analysis)

    stream_varname = gen_var_name(cfg)[0]

    cfg["run_id"] = args.run_id
    cfg["storage"]["run_id"] = cfg["run_id"]
    logger.info(f"Starting run {cfg['run_id']}")

    # Instantiate a storage backend and store the run configuration and task configuration
    store_type = get_storage_object(cfg["storage"])
    store_backend = store_type(cfg["storage"])
    store_backend.store_one({"run_id": cfg['run_id'], "run_config": cfg})

    # TODO: (RMC)  Should this be moved to where cfg updated?
    # (would allow updating channels to process remotely)
    reader = reader_gen(cfg[args.transport], gen_channel_name(cfg["diagnostic"]))
    reader.Open()

    dq = queue.Queue()

    # In a streaming setting, (SST, dataman) attributes can only be accessed after
    # reading the first time step of a variable.
    # Initialize stream_attrs with None and load it in the main loop below.
    stream_attrs = None

    data_model_gen = data_model_generator(cfg["diagnostic"])
    my_preprocessor = preprocessor(executor_pre, cfg)
    my_task_list = tasklist(executor_anl, cfg)

    worker_thread_list = []
    for _ in range(args.num_queue_threads):
        new_worker = threading.Thread(target=consume, args=(dq, my_task_list, my_preprocessor))
        new_worker.start()
        worker_thread_list.append(new_worker)

    logger.info("Starting main loop")
    tic_main = time.perf_counter()
    rx_list = []
    while True:
        stepStatus = reader.BeginStep(timeoutSeconds=5.0)
        if stepStatus:
            # Load attributes
            if stream_attrs is None:
                logger.info("Waiting for attributes")
                stream_attrs = reader.get_attrs("stream_attrs")
                logger.info(f"Got attributes: {stream_attrs}")
            # Read data
            stream_data = reader.Get(stream_varname, save=False)
            # if reader.CurrentStep() in [0, 140]:
            rx_list.append(reader.CurrentStep())

            # Create a datamodel instance from the raw data and push into the queue
            msg = data_model_gen.new_chunk(stream_data, stream_attrs, reader.CurrentStep())
            dq.put_nowait(msg)
            logger.info(f"Published tidx {reader.CurrentStep()}")
            reader.EndStep()
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            break

        if reader.CurrentStep() > 100:
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
