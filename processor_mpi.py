# -*- coding: UTF-8 -*-

"""
This processor implements the one-to-one model using mpi

main loop is based on:
* https://www.roguelynn.com/words/asyncio-true-concurrency/
* Jong's threaded queue code
* Dask codes


To run on an interactive node
srun -n 4 python -m mpi4py.futures processor_mpi.py  --config configs/test_crossphase.json

"""

import os
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import sys
sys.path.append("/global/homes/r/rkube/software/adios2-current/lib64/python3.7/site-packages")


import logging
import logging.config
import random
import string
import queue
import concurrent.futures
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

from readers.reader_mpi import reader_bpfile
from analysis.task_fft import task_fft_scipy
from analysis.tasks_mpi import task_cross_correlation, task_cross_phase, task_cross_power, task_coherence, task_bicoherence, task_skw, task_xspec, task_null


# task_object_dict maps the string-value of the analysis field in the json file
# to an object that defines an appropriate analysis function.
task_object_dict = {"null": task_null,
                    "cross_correlation": task_cross_correlation,
                    "cross_phase": task_cross_phase,
                    "cross_power": task_cross_power,
                    "coherence": task_coherence,
                    "bicoherence": task_bicoherence,
                    "xspec": task_xspec,
                    "skw": task_skw}


@attr.s 
class AdiosMessage:
    """Storage class used to transfer data from Kstar(Dataman) to
    local PoolExecutor"""
    tstep_idx = attr.ib(repr=True)
    data       = attr.ib(repr=False)


cfg = {}


def consume(Q, executor, my_fft, task_list):
    """Executed by a local thread. Dispatch work items from the
    Queue to the PoolExecutor"""

    logger = logging.getLogger('benchmark')
    global cfg

    while True:
        msg = Q.get()
        # If we get our special break message, we exit
        if msg.tstep_idx == None:
            Q.task_done()
            break

        # Step 1) Perform STFT. TODO: We may distribute this among the tasks
        tic_fft = timeit.default_timer()
        fft_data = my_fft.do_fft_local(msg.data)
        toc_fft = timeit.default_timer()
        logger.info(f"tidx={msg.tstep_idx}: FFT took {(toc_fft - tic_fft):6.4f}s")

        # Step 2) Distribute the work via PoolExecutor 
        for task in task_list:
            task.calc_and_store(executor, fft_data, msg.tstep_idx, cfg)

        Q.task_done()
        logger.info(f"Consumed {msg}")


# Procedure that is called to store the data from analysis
# def storage(task, cfg, store_backend):
#     logger = logging.getLogger('benchmark')
#     logger.info(f"Starting storage task. Length of future list: {len(task.futures_list)}")

#     for future in concurrent.futures.as_completed(task.futures_list):
#         future_res, future_info = future.result()
#         logger.info(f"Future complete: {future_info}")
#         store_backend.store(future_res, future_info)


#     logger.info(f"Ending storage task.")


def main():
    # Parse command line arguments and read configuration file
    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/config_null.json')
    parser.add_argument('--benchmark', action="store_true")
    args = parser.parse_args()

    global cfg

    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()

    # create a 6 run id
    cfg['run_id'] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    
    # Load logger configuration from file: 
    # http://zetcode.com/python/logging/
    with open('configs/logger.yaml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)


    # Create a sub-directory in tests_performance if we run in benchmark mode
    if args.benchmark:
        logger = logging.getLogger('benchmark')
        logger.info("Running in benchmark mode")

        tmpdir_name = os.path.join("tests_performance", cfg['run_id'])
        logger.info(f"Runing in benchmark mode. Logging in {tmpdir_name}/performance.log")
        os.mkdir(tmpdir_name)
        
        # Create the file handler
        benchmark_fh = logging.FileHandler(os.path.join(tmpdir_name, 'performance.log'))
        benchmark_formatter = logging.Formatter("%(levelname)s %(asctime)s,%(msecs)d [Process %(process)d %(processName)s %(threadName)s] [%(module)s %(funcName)s]: %(message)s ")
        benchmark_fh.setFormatter(benchmark_formatter)

        logger.addHandler(benchmark_fh)

    else:
        logger = logging.getLogger('simple')


    logger.info(f"Starting run {cfg['run_id']}")

    
    # Instantiate a storage backend and store the run configuration and task configuration
    if cfg['storage']['backend'] == "numpy":
        store_backend = backends.backend_numpy(cfg['storage'])
    elif cfg['storage']['backend'] == "mongo":
        store_backend = backends.backend_mongodb(cfg)    
    elif cfg['storage']['backend'] == "null":
        store_backend = backends.backend_null(cfg['storage'])

    store_backend.store_one({"run_id": cfg['run_id'], "run_config": cfg})


    # Create the FFT task
    cfg["fft_params"]["fsample"] = cfg["ECEI_cfg"]["SampleRate"] * 1e3
    my_fft = task_fft_scipy(10_000, cfg["fft_params"], normalize=True, detrend=True)
    fft_params = my_fft.get_fft_params()

    # Create ADIOS reader object
    reader = reader_bpfile(cfg["shotnr"], cfg["ECEI_cfg"])
    reader.Open(cfg["datapath"])

    # Create a global executor
    #executor = concurrent.futures.ThreadPoolExecutor(max_workers=60)
    executor = MPIPoolExecutor(max_workers=24)

    # Create the task list
    task_list = []
    for task_config in cfg["task_list"]:
        task_list.append(task_object_dict[task_config["analysis"]](task_config, fft_params, cfg["ECEI_cfg"]))
        store_backend.store_metadata(task_config, task_list[-1].get_dispatch_sequence())
        
    dq = queue.Queue()
    msg = None

    tic_main = timeit.default_timer()

    worker = threading.Thread(target=consume, args=(dq, executor, my_fft, task_list))
    worker.start()

    logger.info(f"Starting main loop")
    while True:
        stepStatus = reader.BeginStep()

        if stepStatus:
            # Read data
            stream_data = reader.Get(save=False)
            #tb = reader.gen_timebase()

            # Generate message id and publish is
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data)
            dq.put(msg)
            logger.info(f"Published message {msg}")

        if reader.CurrentStep() >= 20:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            dq.put(AdiosMessage(tstep_idx=None, data=None))
            break

    logger.info("Exiting main loop")
    worker.join()
    logger.info("Workers have joined")
    dq.join()
    logger.info("Queue joined")

    # At this point, all work items have been dispatched. Continue by storing 
    # results of the analysis

    # Spawn len(task_list) workers to store the results of the analysis
    # Use threads since storing is most likely I/O bound:
    # https://timber.io/blog/multiprocessing-vs-multithreading-in-python-what-you-need-to-know/
    # storage_threads = []

    # if cfg['storage']['backend'] == "numpy":
    #     store_backend = backends.backend_numpy(cfg['storage'])
    # elif cfg['storage']['backend'] == "mongo":
    #     store_backend = backends.backend_mongodb(cfg['storage'])
    # elif cfg['storage']['backend'] == "null":
    #     store_backend = backends.backend_null(cfg['storage'])
    
    #     t = threading.Thread(target=storage, args=(task, cfg, store_backend))
    #     t.start()
    #     storage_threads.append(t)

    # logger.info("Joining storage tasks")
    # # Wait for the storage threads to finish
    # for t in storage_threads:
    #     t.join()
    # logger.info("Finished joining storage tasks")

    # Shotdown the executioner
    executor.shutdown(wait=True)

    toc_main = timeit.default_timer()

    logger.info(f"Run {cfg['run_id']} finished in {(toc_main - tic_main):6.4f}s")


if __name__ == "__main__":
    main()


# End of file processor_dask_mp.yp