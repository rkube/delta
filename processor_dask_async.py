# -*- coding: UTF-8 -*-

"""
This processor implements the one-to-one model using an async, non-blocking main loop.

main loop is based on this tutorial: https://www.roguelynn.com/words/asyncio-true-concurrency/


To run: 

On an interactice node:
1. Start the dask scheduler:
python -u $(which dask-scheduler) --scheduler-file $SCRATCH/scheduler.json

2. Start the worker tasks
srun -u -n 10 python -u $(which dask-worker) --scheduler-file $SCRATCH/scheduler.json --nthreads 1 &

3. Launch the dask program:
python -u processor_dask.py
"""

import sys
sys.path.append("/home/rkube/software/adios2-release_25/lib64/python3.7/site-packages")

import asyncio
import logging
import random
import string
import uuid

import attr

import timeit

import json
import argparse

import adios2
from distributed import Client, progress
from backends.backend_numpy import backend_numpy
from readers.reader_one_to_one import reader_bpfile
from analysis.task_fft import task_fft_scipy
from analysis.task_spectral import task_spectral, task_cross_phase, task_cross_power, task_coherence, task_bicoherence, task_xspec, task_cross_correlation, task_skw


# task_object_dict maps the string-value of the analysis field in the json file
# to an object that defines an appropriate analysis function.
task_object_dict = {"cross_phase": task_cross_phase,
                    "cross_power": task_cross_power,
                    "coherence": task_coherence,
                    "bicoherence": task_bicoherence,
                    "xspec": task_xspec,
                    "cross_correlation": task_cross_correlation,
                    "skw": task_skw}


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


async def publish(queue, reader):
    """Simulates an external publisher of messages.
    Args:
        queue (asyncio.Queue): Queue to publish messages to.
        reader: Connection to read bp files
    """


    while True:
        stepStatus = reader.BeginStep()

        if stepStatus:
            print("read: ok")

            # Read data
            stream_data = reader.Get(save=False)
            tb = reader.gen_timebase()

            # Generate message id and publish is
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data)

            asyncio.create_task(queue.put(msg))
            logging.info(f"Published message {msg}")
            # TODO: We need to add a small sleep here so that the consumer catches up
            await asyncio.sleep(0.0)

            if reader.CurrentStep() > 5:
                break

        else:
            print("StepStatus: ", stepStatus)
            break



async def handle_message(msg, dask_client, store_backend, my_fft, task_list, cfg):
    """Performs all work on a data chunk
    -> FFT
    -> Dispatch analysis to dask cluster
    -> Store

    Input:
    ======
    msg
    dask_client
    my_fft
    task_list
    cfg

    Returns:
    ========
    None
    """

    logging.info(f"Handling {msg}")

    tic_fft = timeit.default_timer()
    fft_data = my_fft.do_fft_local(msg.data)
    fft_future = dask_client.scatter(fft_data, broadcast=True)
    toc_fft = timeit.default_timer()
    logging.info(f"FFT + scatter took {(toc_fft-tic_fft):6.4f}s")

    tic_task = timeit.default_timer()
    for task in task_list:
        task.calculate(dask_client, fft_future)
        task.store_data(store_backend, {"tstep": msg.tstep_idx})
    toc_task = timeit.default_timer()
    logging.info(f"Task processing + storage took {(toc_task - tic_task):6.4f}s")



async def consume(queue, dask_client, store_backend, my_fft, task_list, cfg):
    while True:
        # wait for an item from the publisher
        msg = await queue.get()

        # the publisher emits None to indicate that it is done
        if msg is None:
            break

        # Process the message
        asyncio.create_task(handle_message(msg, dask_client, store_backend, my_fft, task_list, cfg))
        logging.info(f"Consumed {msg}")
        # simulate i/o operation using sleep
        await asyncio.sleep(0.1)



def main():

    # Parse command line arguments and read configuration file
    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a dask queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file', default='config_one_to_one_fluctana.json')
    args = parser.parse_args()
    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()

    # Create the FFT task
    cfg["fft_params"]["fsample"] = cfg["ECEI_cfg"]["SampleRate"] * 1e3
    my_fft = task_fft_scipy(10_000, cfg["fft_params"], normalize=True, detrend=True)
    fft_params = my_fft.get_fft_params()

    # Create ADIOS reader object
    reader = reader_bpfile(cfg["shotnr"], cfg["ECEI_cfg"])
    reader.Open(cfg["datapath"])

    # Create dask client
    #dask_client = Client(scheduler_file="/global/cscratch1/sd/rkube/scheduler.json")
    dask_client = Client(scheduler_file="/scratch/gpfs/rkube/dask_work/scheduler.json")
    def add_path():
        import sys
        import numpy as np
        sys.path.append("/home/rkube/repos/delta")
    #dask_client.run(add_path)

    # Create storage backend
    store_backend = backend_numpy("/home/rkube/repos/delta/test_data")

    # Create the task list
    task_list = []
    for task_config in cfg["task_list"]:
        task_list.append(task_object_dict[task_config["analysis"]](task_config, fft_params, cfg["ECEI_cfg"]))
        task_list[-1].store_metadata(store_backend)

    # Create the queue
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()


    print("Starting main loop")
    try:
        loop.create_task(publish(queue, reader))
        #loop.create_task(publish_2(queue))
        loop.create_task(consume(queue, dask_client, store_backend, my_fft, task_list, cfg))
        loop.run_forever()

    except KeyboardInterrupt:
        logging.info("Process interrupted")

    finally:
        loop.close()
        logging.info("Successfully shutdown the delta service.")


if __name__ == "__main__":
    main()

# End of file processor_dask_async.py