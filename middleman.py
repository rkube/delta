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
import time

from streaming.reader_nompi import reader_gen
from streaming.writer_nompi import writer_gen
from analysis.channels import channel_range


@attr.s 
class AdiosMessage:
    """Storage class used to transfer data from Kstar(Dataman) to local PoolExecutor"""
    tstep_idx = attr.ib(repr=True)
    data      = attr.ib(repr=False)


def forward(Q, cfg, timeout):
    """To be executed by a local thread. Pops items from the queue and forwards them."""
    logger = logging.getLogger("middleman")
    writer = writer_gen(cfg["transport_nersc"])
    dummy_data = np.zeros( (192, cfg["transport_nersc"]["chunk_size"]), dtype=np.float64)
    writer.DefineVariable(cfg["transport_nersc"]["channel_range"][0], dummy_data)
    writer.Open()
    logger.info("Starting reader process")
    tx_list = []

    while True:
        try:
            msg = Q.get(timeout=timeout)
        except queue.Empty:
            logger.info("Empty queue after waiting until time-out. Exiting")

        if msg.tstep_idx == None:
            Q.task_done()
            logger.info("Received hangup signal")
            break

        writer.BeginStep()
        writer.put_data(msg.data)
        writer.EndStep()
        time.sleep(0.1)
        tx_list.append(msg.tstep_idx)

        Q.task_done()
        logger.info(f"Consumed tidx={msg.tstep_idx}")
    logger.info(f"Exiting send loop. Transmitted {len(tx_list)} time chunks: {tx_list}")


def main():
    """Reads items from a ADIOS2 connection and forwards them."""

    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/config-middle.json')
    args = parser.parse_args()

    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()
    timeout = 30

    # The middleman uses both a reader and a writer. Each is configured with using their respective section 
    # of the config file. Therefore some keys are duplicated, such as channel_range. Make sure that these
    # items are the same in both sections

    assert(cfg["transport_kstar"]["channel_range"] == cfg["transport_nersc"]["channel_range"])

    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)  
    logger = logging.getLogger('middleman')

    # Create ADIOS reader object
    reader = reader_gen(cfg["transport_kstar"])
    reader.Open()

    dq = queue.Queue()
    msg = None
    worker = threading.Thread(target=forward, args=(dq, cfg, timeout))
    worker.start()

    tic = timeit.default_timer()
    nstep = 0
    rx_list = []
    while True:
        stepStatus = reader.BeginStep()    
        logger.info(f"stepStatus = {stepStatus}, currentStep = {reader.CurrentStep()}")

        if stepStatus:
            if reader.CurrentStep() == 0:
                tic = timeit.default_timer()
            # Read data
            stream_data = reader.Get(channel_range.from_str(cfg["transport_kstar"]["channel_range"][0]), 
                                     save=False)

            rx_list.append(reader.CurrentStep())

            # Generate message id and publish is
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data)
            dq.put_nowait(msg)
            logger.info(f"Published message {msg}")
            reader.EndStep()
            nstep += 1
        else:
            logger.info(f"Exiting: StepStatus={stepStatus}")
            dq.put_nowait(AdiosMessage(tstep_idx=None, data=None))
            break

        last_step = reader.CurrentStep()

    logger.info("Exiting main loop")
    worker.join()
    logger.info("Workers have joined")
    dq.join()
    logger.info("Queue joined")
    toc = timeit.default_timer()
    deltat = toc - tic

    chunk_size = np.prod(stream_data.shape) * stream_data.itemsize / 1024 / 1024
    logger.info(f"Received {len(rx_list)} time chunks: {rx_list}")
    logger.info("")
    logger.info("Summary:")
    logger.info(f"    chunk shape: {stream_data.shape}")
    logger.info(f"    chunk size (MB): {chunk_size:.03f}")
    logger.info(f"    total nstep: {nstep:d}")
    logger.info(f"    total data (MB): {(chunk_size * nstep):03f}")
    logger.info(f"    time (sec): {(deltat):.03f}")
    logger.info(f"    throughput (MB/sec): {(chunk_size * nstep)/(deltat):.03f}")

    logger.info("Finished")


if __name__ == "__main__":
    main()


# End of file middleman.py