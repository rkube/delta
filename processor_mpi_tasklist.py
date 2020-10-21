# -*- coding: UTF-8 -*-

"""
This processor implements the one-to-one model using mpi


To run on an interactive node
srun -n 4 python -m mpi4py.futures processor_mpi_tasklist.py  --config configs/test_all.json

"""

import sys
sys.path.append("/home/rkube/software/gcc/8.3/adios2/lib/python3.8/site-packages")
import os
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import logging
import logging.config
import random
import queue
import threading
import time
import string
import json, yaml
import argparse

import numpy as np

from preprocess.preprocess import preprocessor
from analysis.tasks_mpi import task_list
from streaming.reader_mpi import reader_gen
from data_models.helpers import gen_channel_name, gen_var_name, data_model_generator


def consume(Q, my_task_list, my_preprocessor):
    """Executed by a local thread. Dispatch work items from the
    Queue to the PoolExecutor"""

    logger = logging.getLogger('simple')
    global cfg

    comm  = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger.info("Starting consume")

    while True:
        try:
            msg = Q.get(timeout=60.0)
        except queue.Empty:
            logger.info("Empty queue after waiting until time-out. Exiting")
            break
        # If we get our special break message, we exit
        #if msg.tstep_idx == None:
        #    Q.task_done()
        #    break

        #if(msg.tidx == 1):
        #    np.savez(f"test_data/io_array_tr_s{msg.tidx:04d}.npz", msg.data)

        # TODO: Should there be a general method to a time index from a data chunk?
        logger.info(f"Rank {rank}: Consumed tidx={msg.tb.chunk_idx}. Got data type {type(msg)}")
        my_preprocessor.submit(msg, msg.tb_chunk_idx)
        #my_task_list.submit(msg, msg.tb.chunk_idx)

        Q.task_done()
    logger.info("Task done")


def main():
    comm  = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse command line arguments and read configuration file
    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/config_null.json')
    parser.add_argument('--benchmark', action="store_true")
    args = parser.parse_args()

    global cfg

    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()

    # Load logger configuration from file:
    # http://zetcode.com/python/logging/
    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger = logging.getLogger('simple')

    # Create a global executor
    #executor = concurrent.futures.ThreadPoolExecutor(max_workers=60)
    # PoolExecutor for pre-processing, on-node.
    executor_pre = MPIPoolExecutor(max_workers=4)
    # PoolExecutor for data analysis. off-node
    executor_anl = MPIPoolExecutor(max_workers=24)

    stream_varname = gen_var_name(cfg)[rank]

    cfg["run_id"] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    cfg["run_id"] = "ABC128"
    cfg["storage"]["run_id"] = cfg["run_id"]
    logger.info(f"Starting run {cfg['run_id']}")

    # # Instantiate a storage backend and store the run configuration and task configuration
    # if cfg['storage']['backend'] == "numpy":
    #     store_backend = storage.backend_numpy(cfg['storage'])
    # elif cfg['storage']['backend'] == "mongo":
    #     store_backend = storage.backend_mongodb(cfg["storage"])
    # elif cfg['storage']['backend'] == "null":
    #     store_backend = storage.backend_null(cfg['storage'])
    # else:
    #     raise NameError(f"Unknown storage backend requested: {cfg['storage']['backend']}")

    # store_backend.store_one({"run_id": cfg['run_id'], "run_config": cfg})
    # logger.info(f"Stored one")

    # HARDCODED: create fft_config from current config.json
    fft_config = cfg["preprocess"]["stft"]

    # Create ADIOS reader object
    reader = reader_gen(cfg["transport_nersc"], gen_channel_name(cfg["diagnostic"]))

    my_preprocessor = preprocessor(executor_pre, cfg["preprocess"])
    my_task_list = task_list(executor_anl, cfg["task_list"], cfg["diagnostic"], cfg["storage"])

    dq = queue.Queue()
    msg = None
    data_model_gen = data_model_generator(cfg["diagnostic"])

    tic_main = time.perf_counter()
    workers = []
    for _ in range(4):
        worker = threading.Thread(target=consume, args=(dq, my_task_list, my_preprocessor))
        worker.start()
        workers.append(worker)

    # reader.Open() is blocking until it opens the data file or receives the
    # data stream. Put this right before entering the main loop
    logger.info(f"{rank} Waiting for generator")
    reader.Open()
    logger.info(f"Starting main loop")

    rx_list = []
    while True:
        stepStatus = reader.BeginStep()
        if stepStatus:
            # Read data
            stream_data = reader.Get(stream_varname, save=False)
            rx_list.append(reader.CurrentStep())

            # Create a datamodel instance from the raw data and push into the queue
            msg = data_model_gen.new_chunk(stream_data, reader.CurrentStep())
            dq.put_nowait(msg)
            logger.info(f"Published tidx {reader.CurrentStep()}")
            reader.EndStep()
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            break

        if reader.CurrentStep() > 5:
            logger.info(f"End of the line. Exiting")
            break

    dq.join()
    logger.info("Queue joined")

    logger.info("Exiting main loop")
    for thr in workers:
        thr.join()

    logger.info("Workers have joined")

    # Shotdown the executioner
    executor_anl.shutdown(wait=True)
    executor_pre.shutdown(wait=True)
    #executor.shutdown(wait=True)

    toc_main = time.perf_counter()
    logger.info(f"Run {cfg['run_id']} finished in {(toc_main - tic_main):6.4f}s")
    logger.info(f"Processed {len(rx_list)} time_chunks: {rx_list}")

if __name__ == "__main__":
    main()

# End of file processor_mpi.py
