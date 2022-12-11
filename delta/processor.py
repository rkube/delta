# -*- Encoding: UTF-8 -*-

"""This processor implements the one-to-one model using Ray.

To initiate head and worker nodes, use Slurm bash file

"""


import logging
import logging.config
import time
import json
import yaml
import argparse
import os

import ray
from ray.util.queue import Queue

from preprocess.preprocess import preprocessor
from analysis.task_list import tasklist
from streaming.reader_nompi import reader_gen
from data_models.helpers import gen_channel_name, gen_var_name, data_model_generator
from storage.backend import get_storage_object
 

ray.init(address=os.environ["ip_head"],ignore_reinit_error=True)


@ray.remote
def consume(Q,my_preprocessor,my_task_list,worker):
    """Dispatch work items from the queue on a Ray executor task.

    Each time the chuck is executed on a node worker.
    """
    logger = logging.getLogger('simple')
    logger.info("Starting consume")
    time.sleep(3)
    pid = os.getpid()

    while True:
        try:
            msg = Q.get(timeout=120.0)
        except:
            logger.info("Empty queue after waiting until time-out. Exiting")
            break

        logger.info(f"task {pid}: Consumed: {msg.tb} Got data type {type(msg)}")
        msg = ray.get(my_preprocessor.submit.remote(msg))
        my_task_list.execute.remote(msg)
        

    logger.info("Task done")


def main():
    """Procesess a stream of data chunks on an executor."""

    # Parse command line arguments and read configuration file
    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis" +
                                     "tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file',
                        default='configs/hackathon_test.json')
    parser.add_argument("--transport", type=str,
                        help="Specifies the transport section used to configure the reader",
                        default="transport_rx")
    parser.add_argument("--run_id", type=str,
                        help="Name of database collection to store analysis results in",
                        required=False)

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

    q = Queue()

    # In a streaming setting, (SST, dataman) attributes can only be accessed after
    # reading the first time step of a variable. 
    # Initialize stream_attrs with None and load it in the main loop below.
    stream_attrs = None 

    data_model_gen = data_model_generator(cfg["diagnostic"])
    my_preprocessor = preprocessor.remote(cfg)
    my_task_list = tasklist.remote(cfg)
    
    
    num_workers = int(os.environ["SLURM_NTASKS"]) - 1
    workers = [consume.remote(q,my_preprocessor,my_task_list,j) for j in range(num_workers)]


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
            q.put_nowait(msg)
            logger.info(f"Published tidx {reader.CurrentStep()}")
            reader.EndStep()
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            break
        
        if reader.CurrentStep() > 100:
            break

    logger.info("Queue joined")

    logger.info("Exiting main loop")

    ray.get(workers)
    logger.info("Workers have joined")

    # Shutdown Ray
    ray.shutdown()

    toc_main = time.perf_counter()
    logger.info(f"Run {cfg['run_id']} finished in {(toc_main - tic_main):6.4f}s")
    logger.info(f"Processed {len(rx_list)} time_chunks: {rx_list}")

    
if __name__ == "__main__":
    main()

# End of file processor_mpi.py