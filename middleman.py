# Endocing: UTF-8 -*-

import logging
import logging.config
import queue
import threading

import numpy as np

import attr
import json
import yaml
import argparse

import timeit

from streaming.reader_nompi import reader_gen
from streaming.writer_nompi import writer_gen
from analysis.channels import channel_range


@attr.s 
class AdiosMessage:
    """Storage class used to transfer data from Kstar(Dataman) to local PoolExecutor"""
    tstep_idx = attr.ib(repr=True)
    data      = attr.ib(repr=False)


def forward(Q, cfg):
    """To be executed by a local thread. Pops items from the queue and forwards them."""
    logger = logging.getLogger("simple")
    writer = writer_gen(cfg["transport_tx"])
    dummy_data = np.zeros( (192, cfg["transport_rx"]["chunk_size"]), dtype=np.float64)
    writer.DefineVariable(cfg["transport_tx"]["channel_range"][0], dummy_data)
    writer.Open()
    logger.info("Starting reader process")

    while True:
        msg = Q.get()
        if msg.tstep_idx == None:
            Q.task_done()
            logger.info("Received hangup signal")
            break

        writer.BeginStep()
        writer.put_data(msg.data)
        writer.EndStep()

        Q.task_done()
        logger.info(f"Consumed tidx={msg.tstep_idx} from queue")
    logger.info("Exiting send loop")


def main():
    """Reads items from a ADIOS2 connection and forwards them."""

    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/config-middle.json')
    args = parser.parse_args()

    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()

    # The middleman uses both a reader and a writer. Each is configured with using their respective section 
    # of the config file. Therefore some keys are duplicated, such as channel_range. Make sure that these
    # items are the same in both sections

    assert(cfg["transport_rx"]["channel_range"] == cfg["transport_tx"]["channel_range"])

    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)  
    logger = logging.getLogger('simple')

    # Create ADIOS reader object
    reader = reader_gen(cfg["transport_rx"])
    reader.Open()

    dq = queue.Queue()
    msg = None
    tic_main = timeit.default_timer()
    worker = threading.Thread(target=forward, args=(dq, cfg))
    worker.start()

    while True:
        stepStatus = reader.BeginStep()    
        logger.info(f"stepStatus = {stepStatus}, currentStep = {reader.CurrentStep()}")

        if stepStatus:
            # Read data
            logger.info(f"stepStatus == True")
            stream_data = reader.Get(channel_range.from_str(cfg["transport_rx"]["channel_range"][0]), 
                                     save=True)

            # Generate message id and publish is
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data)
            dq.put_nowait(msg)
            logger.info(f"Published message {msg}")
            reader.EndStep()
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            dq.put_nowait(AdiosMessage(tstep_idx=None, data=None))
            break

        last_step = reader.CurrentStep()

        if last_step > 10:
            logger.info(f"Exiting after 1s0 steps")
            dq.put_nowait(AdiosMessage(tstep_idx=None, data=None))
            break

    logger.info("Exiting main loop")
    worker.join()
    logger.info("Workers have joined")
    dq.join()
    logger.info("Queue joined")


if __name__ == "__main__":
    main()


# End of file middleman.py