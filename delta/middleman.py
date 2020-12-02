# Endocing: UTF-8 -*-


"""Receives data from generator and forwards them to processor."""

from mpi4py import MPI

import logging
import logging.config
import queue
import threading
import attr
import json
import yaml
import argparse

from streaming.writers import writer_gen
from streaming.reader_mpi import reader_gen
from data_models.helpers import gen_channel_name, gen_var_name


@attr.s
class AdiosMessage:
    """Dummy replacement for data_chunk class."""
    tstep_idx = attr.ib(repr=True)
    data = attr.ib(repr=False)
    attrs = attr.ib(repr=False)


def forward(Q, cfg, args, timeout):
    """To be executed by a local thread. Pops items from the queue and forwards them."""
    global comm, rank
    logger = logging.getLogger("middleman")
    logger.info(f"Worker: Creating writer_gen: engine={cfg[args.transport_tx]['engine']}")

    # suffix = ""  # if not args.debug else '-MM'
    ch_name = gen_channel_name(cfg["diagnostic"])
    writer = writer_gen(cfg[args.transport_tx], ch_name)
    logger.info(f"Worker: Streaming channel name = {ch_name}")

    tx_list = []
    is_first = True
    while True:
        # msg = None
        try:
            msg = Q.get(timeout=timeout)
            logger.info(f"Worker: Receiving from Queue: {msg} - {msg.data.shape}, {msg.data.dtype}")
            if is_first:
                writer.DefineVariable(gen_var_name(cfg)[rank], msg.data.shape, msg.data.dtype)
                #if msg.attrs is not None:
                writer.DefineAttributes("stream_attrs", msg.attrs)
                logger.info(f"Worker: Defining stream_attrs for forwarded stream: {msg.attrs}")
                writer.Open()
                logger.info("Worker: Starting forwarding process")
                is_first = False
        except queue.Empty:
            logger.info("Worker: Empty queue after waiting until time-out. Exiting")
            break

        logger.info(f"Worker Forwarding chunk {msg.tstep_idx}. Data = {msg.data.shape}")
        writer.BeginStep()
        writer.put_data(msg)
        writer.EndStep()
        time.sleep(0.1)
        logger.info(f"Worker: Done writing chunk {msg.tstep_idx}.")
        tx_list.append(msg.tstep_idx)

        Q.task_done()
        logger.info(f"Consumed tidx={msg.tstep_idx}")

    logger.info(f"Worker: Exiting send loop. Transmitted {len(tx_list)} time chunks: {tx_list}")
    logger.info(writer.transfer_stats())


def main():
    """Reads items from a ADIOS2 connection and forwards them."""
    global comm, rank, args
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Receive data and dispatch" +
                                     "analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file',
                        default='configs/config-middle.json')
    parser.add_argument("--transport_rx", help="Specifies the name of the transport section that is used to configure the reader",
                        default="transport_rx")
    parser.add_argument("--transport_tx", help="Specifies the name of the transport section that is used to configure the writer",
                        default="transport_tx")
    args = parser.parse_args()

    with open(args.config, "r") as df:
        cfg = json.load(df)
    timeout = 5

    # The middleman uses both a reader and a writer. Each is configured with using
    # their respective section of the config file. Therefore some keys are duplicated,
    # such as channel_range. Make sure that these items are the same in both sections

    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger = logging.getLogger('middleman')

    # Create ADIOS reader object
    reader = reader_gen(cfg[args.transport_rx], gen_channel_name(cfg["diagnostic"]))
    reader.Open()
    stream_attrs = None
    stream_varname = gen_var_name(cfg)[rank]
    logger.info(f"Main: Stream varname: {stream_varname}")

    dq = queue.Queue()
    msg = None
    worker = threading.Thread(target=forward, args=(dq, cfg, args, timeout))
    worker.start()

    rx_list = []
    stream_data = None
    while True:
        stepStatus = reader.BeginStep()
        logger.info(f"Main: stepStatus = {stepStatus}, currentStep = {reader.CurrentStep()}")
        if stepStatus:
            # Read data
            stream_data = reader.Get(stream_varname, save=False)
            if stream_attrs is None:
                if reader.InquireAttribute("stream_attrs"):
                    stream_attrs = reader.get_attrs("stream_attrs")

            stream_data = reader.Get(stream_varname, save=False)
            rx_list.append(reader.CurrentStep())

            # Generate message id and publish is
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data, attrs=stream_attrs)
            dq.put_nowait(msg)
            logger.info(f"Main: Published message {msg}")
            reader.EndStep()
        else:
            logger.info(f"Main: Exiting: StepStatus={stepStatus}")
            break

        # last_step = reader.CurrentStep()

    logger.info("Main: Exiting main loop")
    worker.join()
    logger.info("Main: Workers have joined")
    dq.join()
    logger.info("Main: Queue joined")
    logger.info("Main: Finished")


if __name__ == "__main__":
    main()

# End of file middleman.py
