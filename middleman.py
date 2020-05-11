# Endocing: UTF-8 -*-

import logging
import queue
import threading

import json
import argparse

from streaming.reader_nompi import reader_gen
from streaming.writer_nompi import writer_gen

def forward(Q, cfg):
    """To be executed by a local thread. Pops items from the queue and forwards them."""
    logger = logging.getLogger("simple")
    writer = writer_gen(cfg["trasport_tx"])

    while True:
        msg = Q.get()
        if msg.tstep_idx == None:
            Q.task_done()
            break

        writer.BeginSte()
        writer.put_data(msg.data)
        writer.EndStep()


 

def main():
    """Reads items from a ADIOS2 connection and forwards them."""

    parser = argparse.ArgumentParser(description="Receive data and dispatch analysis tasks to a mpi queue")
    parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/config_middleman.json')
    args = parser.parse_args()

    with open(args.config, "r") as df:
        cfg = json.load(df)
        df.close()

    with open("configs/logger.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)  
    logger = logging.getLogger('simple')

    # Create ADIOS reader object
    reader = reader_gen(cfg["transport_rx"])


    dq = queue.Queue()
    msg = None
    tic_main = timeit.default_timer()
    worker = threading.Thread(target=forward, args=(dq, cfg))
    worker.start()

    while True:
        stepStatus = reader.BeginStep()
    
        logger.info(f"stepStatus = {stepStatus}")
        #if last_step == reader.CurrentStep():
        #    continue
        logger.info(f"currentStep = {reader.CurrentStep()}")
        if stepStatus:
            # Read data
            logger.info(f"stepStatus == True")
            stream_data = reader.Get(adios2_varname, save=True)

            # Generate message id and publish is
            msg = AdiosMessage(tstep_idx=reader.CurrentStep(), data=stream_data)
            #dq.put_nowait(msg)
            logger.info(f"Published message {msg}")
            reader.EndStep()
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


if __name__ == "__main__":
    main()


# End of file middleman.py