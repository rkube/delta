# Endocing: UTF-8 -*-


"""Receives data from generator and forwards them to processor."""

from mpi4py import MPI

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

from sources.dataloader import get_loader
from streaming.writers import writer_gen
from streaming.reader_mpi import reader_gen
from data_models.helpers import gen_channel_name, gen_var_name

@attr.s
class AdiosMessage:
    """Storage class used to transfer data from Kstar(Dataman) to local PoolExecuto."""
    tstep_idx = attr.ib(repr=True)
    # (2020/11) jyc: Good place to wrap numpy data with ecei_chunk?
    data = attr.ib(repr=False)


def forward(Q, cfg, timeout):
    global comm, rank, args
    """To be executed by a local thread. Pops items from the queue and forwards them."""
    logger = logging.getLogger("middleman")
    
    # Instantiate a dataloader
    dataloader = get_loader(cfg)
    logger.info(f"Creating writer_gen: engine={cfg['transport_nersc']['engine']}")

    suffix = '' if not args.debug else '-MM'
    writer = writer_gen(cfg["transport_nersc"], gen_channel_name(cfg["diagnostic"])+suffix)
    logger.info(f"Streaming channel name = {gen_channel_name(cfg['diagnostic'])}")
    # Give the writer hints on what kind of data to transfer

    writer.DefineVariable(gen_var_name(cfg)[rank],
                        dataloader.get_chunk_shape(),
                        dataloader.dtype)
    writer.Open()
    logger.info("Starting forwarding process")

    tx_list = []

    while True:
        msg = None
        try:
            msg = Q.get(timeout=timeout)
        except queue.Empty:
            logger.info("Empty queue after waiting until time-out. Exiting")

        if msg.tstep_idx is None:
            Q.task_done()
            logger.info("Received hangup signal")
            break

        writer.BeginStep()
        # (2020/11) jyc: This is a hack. Need to wrap as ecei_chunk object
        writer.put_data_np(msg.data, {"tidx": msg.tstep_idx})
        writer.EndStep()
        time.sleep(0.1)
        tx_list.append(msg.tstep_idx)

        Q.task_done()
        logger.info(f"Consumed tidx={msg.tstep_idx}")
    logger.info(f"Exiting send loop. Transmitted {len(tx_list)} time chunks: {tx_list}")


def main():
    """Reads items from a ADIOS2 connection and forwards them."""
    global comm, rank, args
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Receive data and dispatch" +
                                     "analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file',
                        default='configs/config-middle.json')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config, "r") as df:
        cfg = json.load(df)
    timeout = 30

    # The middleman uses both a reader and a writer. Each is configured with using
    # their respective section of the config file. Therefore some keys are duplicated,
    # such as channel_range. Make sure that these items are the same in both sections

    # assert(cfg["transport_kstar"]["channel_range"] == cfg["transport_nersc"]["channel_range"])

    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger = logging.getLogger('middleman')

    # Create ADIOS reader object
    reader = reader_gen(cfg["transport_nersc"], gen_channel_name(cfg["diagnostic"]))
    reader.Open()
    stream_varname = gen_var_name(cfg)[rank]
    logger.info(f"Stream varname: {stream_varname}")

    dq = queue.Queue()
    msg = None
    worker = threading.Thread(target=forward, args=(dq, cfg, timeout))
    worker.start()

    tic = timeit.default_timer()
    nstep = 0
    rx_list = []
    stream_data = None
    while True:
        stepStatus = reader.BeginStep()
        logger.info(f"stepStatus = {stepStatus}, currentStep = {reader.CurrentStep()}")

        if stepStatus:
            if reader.CurrentStep() == 0:
                tic = timeit.default_timer()
            # Read data
            stream_data = reader.Get(stream_varname, save=False)

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

        # last_step = reader.CurrentStep()

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
