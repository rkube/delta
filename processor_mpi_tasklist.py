# -*- coding: UTF-8 -*-

"""
This processor implements the one-to-one model using mpi

main loop is based on:
* https://www.roguelynn.com/words/asyncio-true-concurrency/
* Jong's threaded queue code


To run on an interactive node
srun -n 4 python -m mpi4py.futures processor_mpi_tasklist.py  --config configs/test_all.json

"""

import os
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import logging
import logging.config
import random
import string
import queue
import threading

import numpy as np
import attr
import timeit
import datetime

import json
import yaml
import argparse
import adios2

import backends

from streaming.reader_mpi import reader_gen

#from analysis.task_fft import task_fft_scipy
from analysis.tasks_mpi import task_list_spectral
from analysis.channels import channel_range


@attr.s 
class AdiosMessage:
    """Storage class used to transfer data from Kstar(Dataman) to local PoolExecutor"""
    tstep_idx = attr.ib(repr=True)
    data      = attr.ib(repr=False)

cfg = {}


# class ConsumeThread(threading.Thread):
#     def __init__(self, Q, executor, task_list, cfg):
#         """ constructor, setting initial variables """
#         self.Q = Q
#         self.executor = executor 
#         self.task_list = task_list 
#         self.logger = logging.getLogger("simple")

#         self.logger.info("Constructing thread")

#         self.cfg = cfg
#         self.cfg["fft_params"]["fsample"] = self.cfg["ECEI_cfg"]["SampleRate"] * 1e3
#         self.my_fft = task_fft_scipy(10_000, self.cfg["fft_params"], normalize=True, detrend=True)

#         self._interrupt = threading.Event()
#         threading.Thread.__init__(self)
        

#     def interrupt(self):
#         self._interrupt.set()
        
#     def run(self):
#         #logger = logging.getLogger('simple')
#         global cfg

#         #comm  = MPI.COMM_WORLD
#         #rank = comm.Get_rank()
#         #size = comm.Get_size()

#         # Create the FFT task
#         self.logger.info("Starting thread")

#         while True:
#             try:
#                 msg = Q.get(timeout=5.0)
#             except queue.Empty:
#                 self.logger.info("Queue is empty. Exiting")
#                 break

#             # If we get our special break message, we exit
#             if msg.tstep_idx == None:
#                 Q.task_done()
#                 break

#             # Step 1) Perform STFT. TODO: We may distribute this among the tasks
#             tic_fft = timeit.default_timer()
#             fft_data = self.my_fft.do_fft_local(msg.data)
#             toc_fft = timeit.default_timer()
#             logger.info(f"tidx={msg.tstep_idx}: FFT took {(toc_fft - tic_fft):6.4f}s")

#             # Step 2) Distribute tasks to the executor 
#             for task in self.task_list:
#                 task.submit(self.executor, fft_data, msg.tstep_idx, self.cfg)

#             Q.task_done()


def consume(Q, task_list):
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
            msg = Q.get(timeout=5.0)
        except queue.Empty:
            logger.info("Empty queue after waiting until time-out. Exiting")
            break
        # If we get our special break message, we exit
        if msg.tstep_idx == None:
            Q.task_done()
            break

        #if(msg.tidx == 1):
        #    np.savez(f"test_data/io_array_tr_s{msg.tidx:04d}.npz", msg.data)

        logger.info(f"Rank {rank}: Consumed tidx={msg.tstep_idx}")
        task_list.submit(msg.data, msg.tstep_idx)

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
    executor_fft = MPIPoolExecutor(max_workers=16)
    executor_anl = MPIPoolExecutor(max_workers=16)
    #executor = MPIPoolExecutor(max_workers=120)

    adios2_varname = channel_range.from_str(cfg["transport_nersc"]["channel_range"][0])

    
    cfg["run_id"] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    cfg["run_id"] = "ABC128"
    cfg["storage"]["run_id"] = cfg["run_id"]
    logger.info(f"Starting run {cfg['run_id']}")

    # Instantiate a storage backend and store the run configuration and task configuration
    if cfg['storage']['backend'] == "numpy":
        store_backend = backends.backend_numpy(cfg['storage'])
    elif cfg['storage']['backend'] == "mongo":
        store_backend = backends.backend_mongodb(cfg["storage"])
    elif cfg['storage']['backend'] == "null":
        store_backend = backends.backend_null(cfg['storage'])
    else:
        raise NameError(f"Unknown storage backend requested: {cfg['storage']['backend']}")

    store_backend.store_one({"run_id": cfg['run_id'], "run_config": cfg})
    logger.info(f"Stored one")

    # Create ADIOS reader object
    reader = reader_gen(cfg["transport_nersc"])
    task_list = task_list_spectral(executor_anl, executor_fft, cfg["task_list"], cfg["fft_params"], cfg["ECEI_cfg"], cfg["storage"])
    #task_list = task_list_spectral(executor, cfg["task_list"], cfg["fft_params"], cfg["ECEI_cfg"], cfg["storage"])

    dq = queue.Queue()
    msg = None#

    tic_main = timeit.default_timer()
    workers = []
    for _ in range(4):
        worker = threading.Thread(target=consume, args=(dq, task_list))
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
            stream_data = reader.Get(adios2_varname, save=False)
            rx_list.append(reader.CurrentStep())

            # Generate message id and publish 
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data)
            dq.put_nowait(msg)
            logger.info(f"Published tidx {msg.tstep_idx}")
            reader.EndStep()
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            break

        if reader.CurrentStep() > 100:
            break

    dq.join()
    logger.info("Queue joined")

    logger.info("Exiting main loop")
    for thr in workers:
        thr.join()

    logger.info("Workers have joined")

    # Shotdown the executioner
    executor_anl.shutdown(wait=True)
    executor_fft.shutdown(wait=True)
    #executor.shutdown(wait=True)

    toc_main = timeit.default_timer()
    logger.info(f"Run {cfg['run_id']} finished in {(toc_main - tic_main):6.4f}s")
    logger.info(f"Processed {len(rx_list)} time_chunks: {rx_list}")

if __name__ == "__main__":
    main()

# End of file processor_mpi.py
