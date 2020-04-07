# -*- Encoding: UTF-8 -*-

from mpi4py import MPI 
import numpy as np
import adios2
import argparse
import json
import yaml
import logging, logging.config

from analysis.channels import channel_range
from streaming.readers import reader_dataman


"""
Author: Ralph Kube

A simple processor model than consumes data from the helloDatManWriter.py example
from the adios2 examples.
Use the channel name HelloDataMan
""" 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

config_str = """{
    "shotnr": 18431,
    "datapath": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/018431/",
    "nstep": 10,
    "channel_range": ["L0101-2408"],
    "transport":
    {
      "chunk_size": 10000,
      "engine": "dataman",
      "params":
      {
        "IPAddress": "128.55.205.18", 
        "Timeout": "20",
        "Port": "12306",
        "OneToOneMode": "TRUE",
        "OpenTimeoutSecs": "30"
      }
    }
}"""

parser = argparse.ArgumentParser(description="Send KSTAR data using ADIOS2")
parser.add_argument('--config', type=str, help='Lists the configuration file', default='configs/test_generator.json')
args = parser.parse_args()

#with open(args.config, "r") as df:
#    cfg = json.load(df)
cfg = json.loads(config_str)

with open('configs/logger.yaml', 'r') as f:
    log_cfg = yaml.safe_load(f.read())
logging.config.dictConfig(log_cfg)

logger = logging.getLogger("simple")

logger.info(f"{cfg['channel_range']}")

gen_id = 1_000 * rank

my_channel_range = channel_range.from_str(cfg["channel_range"][rank])
logger.info(f"rank: {rank} config: {cfg}")

reader = reader_dataman(cfg)
logger.info("Waiting")
reader.Open()

step = 0

while(True):
    stepStatus = reader.BeginStep()
    logger.info(stepStatus)
    if stepStatus == adios2.StepStatus.OK:
        channel_data = reader.get_data("FloatArray")
        reader.EndStep()
        logger.info(f"rank {rank:d}: Step  {reader.CurrentStep():04d}  data = {channel_data}")
    else:
        logger.info(f">>> receiver {rank:d}: End of stream")
        break

    # Recover channel data 
    #channel_data = channel_data.reshape((num_channels, channel_data.size // num_channels))

    logger.info(f">>> Step begins ... {step}")
    ## jyc: this is just for testing. This is a place to run analysis if we want.
    ##executor.submit(perform_analysis, channel_data, step)

    ## Save data in a queue so that a workder thead will fetch and save concurrently.
    #queue_list[step%num_analysis].put((channel_data, step))
    step += 1

    if step > 10:
        break


# End of file processor_simple.py